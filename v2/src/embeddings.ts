import { createHash } from "node:crypto";
import { z } from "zod";
import { Gate, readBounded } from "./limits";
import { RagError } from "./contracts";

export interface EmbeddingProvider {
  readonly spaceId: string;
  readonly dimensions: number;
  readonly maxInputBytes: number;
  embedQuery(text: string, signal: AbortSignal): Promise<readonly number[]>;
  embedDocuments(
    texts: readonly string[],
    signal: AbortSignal,
  ): Promise<readonly (readonly number[])[]>;
}
export function validateVectors(
  vectors: readonly (readonly number[])[],
  count: number,
  dimensions: number,
): void {
  if (
    vectors.length !== count ||
    vectors.some(
      (vector) =>
        vector.length !== dimensions ||
        vector.some((value) => !Number.isFinite(value)) ||
        !vector.some((value) => value !== 0),
    )
  )
    throw new RagError("INVALID_EMBEDDINGS", 503);
}
const responseSchema = z.object({
  data: z.array(
    z.object({
      index: z.number().int().nonnegative(),
      embedding: z.array(z.number().finite()),
    }),
  ),
});

export function openAICompatible(options: {
  endpoint: string;
  apiKey: string;
  model: string;
  dimensions: number;
  concurrency?: number;
}): EmbeddingProvider {
  const url = new URL(options.endpoint);
  if (
    url.protocol !== "https:" &&
    !(
      url.protocol === "http:" &&
      ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)
    )
  )
    throw new Error("EMBEDDING_ENDPOINT_REQUIRES_TLS");
  if (url.username || url.password || url.hash || url.search)
    throw new Error("INVALID_EMBEDDING_ENDPOINT");
  const spaceId = createHash("sha256")
    .update(
      JSON.stringify([
        url.href,
        options.model,
        options.dimensions,
        "raw-query/title-section-document-budgeted-v2",
      ]),
    )
    .digest("hex");
  const gate = new Gate(options.concurrency ?? 4, 16);
  const embed = (texts: readonly string[], signal: AbortSignal) =>
    gate.run(signal, async () => {
      if (
        !texts.length ||
        texts.length > 64 ||
        texts.some((text) => !text.trim() || Buffer.byteLength(text) > 8191)
      )
        throw new RagError("EMBEDDING_INPUT_LIMIT", 422);
      for (let attempt = 0; attempt < 2; attempt++) {
        let response: Response;
        try {
          response = await fetch(url, {
            method: "POST",
            redirect: "error",
            signal,
            headers: {
              Authorization: `Bearer ${options.apiKey}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              model: options.model,
              dimensions: options.dimensions,
              input: texts,
            }),
          });
        } catch (error) {
          signal.throwIfAborted();
          if (attempt || !(error instanceof TypeError))
            throw new RagError("EMBEDDING_UNAVAILABLE", 503);
          await Bun.sleep(100);
          continue;
        }
        if (response.status === 429 || response.status >= 500) {
          await response.body?.cancel();
          if (attempt) throw new RagError("EMBEDDING_UNAVAILABLE", 503);
          await Bun.sleep(100);
          signal.throwIfAborted();
          continue;
        }
        if (!response.ok) {
          await response.body?.cancel();
          throw new RagError("EMBEDDING_REJECTED", 503);
        }
        let parsed: z.infer<typeof responseSchema>;
        try {
          parsed = responseSchema.parse(
            JSON.parse(
              new TextDecoder("utf-8", { fatal: true }).decode(
                await readBounded(
                  response.body,
                  texts.length * options.dimensions * 32 + 65536,
                  signal,
                ),
              ),
            ),
          );
        } catch {
          signal.throwIfAborted();
          throw new RagError("INVALID_EMBEDDINGS", 503);
        }
        const vectors: number[][] = Array(texts.length);
        for (const entry of parsed.data) {
          if (entry.index >= texts.length || vectors[entry.index])
            throw new RagError("INVALID_EMBEDDINGS", 503);
          vectors[entry.index] = entry.embedding;
        }
        if (
          parsed.data.length !== texts.length ||
          vectors.some((vector) => !vector)
        )
          throw new RagError("INVALID_EMBEDDINGS", 503);
        validateVectors(vectors, texts.length, options.dimensions);
        return vectors;
      }
      throw new RagError("EMBEDDING_UNAVAILABLE", 503);
    });
  return {
    spaceId,
    dimensions: options.dimensions,
    maxInputBytes: 8191,
    embedQuery: async (text, signal) => (await embed([text], signal))[0]!,
    embedDocuments: embed,
  };
}

export function deterministicProvider(dimensions = 64): EmbeddingProvider {
  const embed = (text: string) => {
    const vector = Array<number>(dimensions).fill(0);
    for (const token of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
      const hash = createHash("sha256").update(token).digest();
      const index = hash.readUInt32LE(0) % dimensions;
      vector[index] = vector[index]! + 1;
    }
    if (!vector.some(Boolean)) vector[0] = 1;
    return vector;
  };
  return {
    spaceId: `test-only-token-hash-${dimensions}-v2`,
    dimensions,
    maxInputBytes: 8191,
    async embedQuery(text, signal) {
      signal.throwIfAborted();
      return embed(text);
    },
    async embedDocuments(texts, signal) {
      signal.throwIfAborted();
      return texts.map(embed);
    },
  };
}

export class QueryEmbeddings {
  private readonly cache = new Map<
    string,
    { expires: number; vector: readonly number[] }
  >();
  private readonly pending = new Map<string, Promise<readonly number[]>>();
  constructor(
    private readonly provider: EmbeddingProvider,
    private readonly maxEntries = 128,
  ) {}
  async get(text: string, signal: AbortSignal): Promise<readonly number[]> {
    signal.throwIfAborted();
    const key = createHash("sha256")
      .update(`${this.provider.spaceId}\0${text}`)
      .digest("hex");
    const cached = this.cache.get(key);
    if (cached && cached.expires > Date.now()) {
      this.cache.delete(key);
      this.cache.set(key, cached);
      return cached.vector;
    }
    this.cache.delete(key);
    let task = this.pending.get(key);
    if (!task) {
      if (this.pending.size >= 16) throw new RagError("OVERLOADED", 503);
      task = this.provider
        .embedQuery(text, AbortSignal.timeout(10000))
        .then((vector) => {
          validateVectors([vector], 1, this.provider.dimensions);
          this.cache.set(key, { vector, expires: Date.now() + 60000 });
          while (this.cache.size > this.maxEntries)
            this.cache.delete(this.cache.keys().next().value!);
          return vector;
        })
        .finally(() => this.pending.delete(key));
      this.pending.set(key, task);
    }
    return new Promise((resolve, reject) => {
      const abort = () => reject(signal.reason);
      signal.addEventListener("abort", abort, { once: true });
      task
        .then(resolve, reject)
        .finally(() => signal.removeEventListener("abort", abort));
    });
  }
}
