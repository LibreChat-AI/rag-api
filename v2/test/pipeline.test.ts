import { expect, test } from "bun:test";
import { Pipeline } from "../src/pipeline";
import { ingestSchema, searchSchema } from "../src/contracts";
import { MemoryStore, provider, signal } from "./helpers";

const scope = { tenantId: "tenant-a", namespaceId: "library-a" };
const input = (text = "alpha document") =>
  ingestSchema.parse({
    title: "Example",
    segments: [{ kind: "page", index: 1, text }],
  });

test("publishes only after complete bounded batches and carries ordered source locations", async () => {
  const store = new MemoryStore();
  const pipeline = new Pipeline(store, provider);
  const doc = await pipeline.ingest(
    scope,
    "file-a",
    "user",
    "key",
    input("alpha ".repeat(18000)),
    signal(),
  );
  expect(store.events.at(-1)).toBe("publish");
  expect(
    store.events.filter((event) => event === "insert").length,
  ).toBeGreaterThan(1);
  expect(doc.chunkCount).toBe(store.chunks.length);
  expect(store.chunks.map((chunk) => chunk.index)).toEqual(
    Array.from({ length: doc.chunkCount }, (_, index) => index),
  );
  expect(
    store.chunks.every(
      (chunk) => chunk.page === 1 && chunk.embedding.length === 64,
    ),
  ).toBe(true);
  expect(BigInt(doc.version)).toBeGreaterThan(BigInt(Number.MAX_SAFE_INTEGER));
});
test("partial replacement failure leaves the previous generation live and staged chunks invisible", async () => {
  const store = new MemoryStore();
  const pipeline = new Pipeline(store, provider);
  const old = await pipeline.ingest(
    scope,
    "file-a",
    "user",
    "old",
    input(),
    signal(),
  );
  store.failBatch = 3;
  await expect(
    pipeline.ingest(
      scope,
      "file-a",
      "user",
      "new",
      input("replacement ".repeat(12000)),
      signal(),
    ),
  ).rejects.toThrow();
  expect((await store.get(scope, "file-a"))!.generation).toBe(old.generation);
  const hits = await store.search(
    [scope],
    await provider.embedQuery("alpha", signal()),
    provider.spaceId,
    searchSchema.parse({
      query: "alpha",
      namespaces: [{ namespaceId: "library-a" }],
    }),
  );
  expect(hits.every((hit) => hit.generation === old.generation)).toBe(true);
});
test("durable idempotency prevents duplicate ingestion and rejects a reused key with another body", async () => {
  const store = new MemoryStore();
  const pipeline = new Pipeline(store, provider);
  const doc = await pipeline.ingest(
    scope,
    "file-a",
    "user",
    "key",
    input(),
    signal(),
  );
  const replay = await new Pipeline(store, provider).ingest(
    scope,
    "file-a",
    "user",
    "key",
    input(),
    signal(),
  );
  expect(replay).toEqual(doc);
  expect(store.chunks).toHaveLength(1);
  await expect(
    pipeline.ingest(
      scope,
      "file-a",
      "user",
      "key",
      input("different"),
      signal(),
    ),
  ).rejects.toMatchObject({ code: "IDEMPOTENCY_CONFLICT" });
});
test("recovers publication acknowledgment loss without re-embedding", async () => {
  const store = new MemoryStore();
  const pipeline = new Pipeline(store, provider);
  store.failRecord = true;
  await expect(
    pipeline.ingest(scope, "file-a", "user", "key", input(), signal()),
  ).rejects.toThrow();
  store.failRecord = false;
  await new Pipeline(store, provider).ingest(
    scope,
    "file-a",
    "user",
    "key",
    input(),
    signal(),
  );
  expect(store.chunks).toHaveLength(1);
});
test("same-document writers and deletes cannot race, and failed preconditions do not insert", async () => {
  const store = new MemoryStore();
  const pipeline = new Pipeline(store, provider);
  const a = pipeline.ingest(scope, "file-a", "user", "one", input(), signal());
  await expect(
    pipeline.delete(scope, "file-a", "user", "delete", signal()),
  ).rejects.toMatchObject({ code: "DOCUMENT_BUSY" });
  const doc = await a;
  await expect(
    pipeline.ingest(
      scope,
      "file-a",
      "user",
      "two",
      { ...input(), ifMatch: "f".repeat(32) },
      signal(),
    ),
  ).rejects.toMatchObject({ code: "PRECONDITION_FAILED" });
  expect(store.chunks).toHaveLength(1);
  await pipeline.delete(scope, "file-a", "user", "delete", signal());
  expect((await store.get(scope, "file-a"))!.state).toBe("deleted");
  expect((await store.get(scope, "file-a"))!.generation).toBe(doc.generation);
});
test("invalid or cancelled embedding work never publishes", async () => {
  const store = new MemoryStore();
  const invalid = new Pipeline(store, {
    ...provider,
    async embedDocuments() {
      return [[NaN]];
    },
  });
  await expect(
    invalid.ingest(scope, "file-a", "user", "key", input(), signal()),
  ).rejects.toMatchObject({ code: "INVALID_EMBEDDINGS" });
  expect(store.documents.size).toBe(0);
  const controller = new AbortController();
  controller.abort();
  await expect(
    new Pipeline(store, provider).ingest(
      scope,
      "file-a",
      "user",
      "key",
      input(),
      controller.signal,
    ),
  ).rejects.toThrow();
  expect(store.documents.size).toBe(0);
});

test("embeds the next bounded batch while the preceding insert is in flight", async () => {
  const store = new MemoryStore();
  let release = () => {};
  let embedded = 0;
  const insert = store.insert.bind(store);
  store.insert = async (rows) => {
    if (rows[0]!.index === 0)
      await new Promise<void>((resolve) => {
        release = resolve;
      });
    await insert(rows);
  };
  const pipeline = new Pipeline(store, {
    ...provider,
    async embedDocuments(texts, sig) {
      embedded++;
      return provider.embedDocuments(texts, sig);
    },
  });
  const task = pipeline.ingest(
    scope,
    "file-a",
    "user",
    "overlap",
    input("alpha ".repeat(18000)),
    signal(),
  );
  for (let tries = 0; embedded < 2 && tries < 100; tries++) await Bun.sleep(1);
  expect(embedded).toBe(2);
  expect(store.documents.size).toBe(0);
  release();
  await task;
});
test("persisted version floors survive clock skew and provenance is server-owned", async () => {
  const store = new MemoryStore();
  const original = await new Pipeline(store, provider).ingest(
    scope,
    "file-a",
    "user",
    "first",
    input(),
    signal(),
  );
  original.version = "9999999999999999999";
  const next = await new Pipeline(store, provider).ingest(
    scope,
    "file-a",
    "service-worker",
    "second",
    input("replacement"),
    signal(),
    { actorKind: "service", sourceClass: "extracted" },
  );
  expect(BigInt(next.version)).toBeGreaterThan(BigInt(original.version));
  expect(next.actor).toBe("service:service-worker");
  expect(next.sourceClass).toBe("extracted");
  expect(store.chunks.at(-1)!.actor).toBe(next.actor);
});

test("a lost receipt cannot make a late retry resurrect an older revision", async () => {
  const store = new MemoryStore();
  const pipeline = new Pipeline(store, provider);
  store.failRecord = true;
  await expect(
    pipeline.ingest(scope, "file-a", "user", "old", input("old"), signal()),
  ).rejects.toThrow();
  const original = (await store.get(scope, "file-a"))!;
  store.failRecord = false;
  const updated = await pipeline.ingest(
    scope,
    "file-a",
    "user",
    "new",
    input("new"),
    signal(),
  );
  const replay = await new Pipeline(store, provider).ingest(
    scope,
    "file-a",
    "user",
    "old",
    input("old"),
    signal(),
  );
  expect(replay.generation).toBe(original.generation);
  expect((await store.get(scope, "file-a"))!.generation).toBe(
    updated.generation,
  );
});

test("a stale ifMatch cannot resurrect a deleted generation, but deliberate recreation still works", async () => {
  const store = new MemoryStore();
  const pipeline = new Pipeline(store, provider);
  const original = await pipeline.ingest(
    scope,
    "file-a",
    "user",
    "original",
    input(),
    signal(),
  );
  await pipeline.delete(scope, "file-a", "user", "delete", signal());
  const inserts = store.events.filter((event) => event === "insert").length;
  await expect(
    new Pipeline(store, provider).ingest(
      scope,
      "file-a",
      "user",
      "stale-edit",
      { ...input("changed"), ifMatch: original.generation },
      signal(),
    ),
  ).rejects.toMatchObject({ code: "PRECONDITION_FAILED" });
  expect(store.events.filter((event) => event === "insert")).toHaveLength(
    inserts,
  );
  expect((await store.get(scope, "file-a"))!.state).toBe("deleted");
  const recreated = await pipeline.ingest(
    scope,
    "file-a",
    "user",
    "recreate",
    input("new"),
    signal(),
  );
  expect(recreated.state).toBe("ready");
  expect(recreated.generation).not.toBe(original.generation);
  const updated = await pipeline.ingest(
    scope,
    "file-a",
    "user",
    "live-edit",
    { ...input("edited"), ifMatch: recreated.generation },
    signal(),
  );
  expect(updated.generation).not.toBe(recreated.generation);
});

test("excess Markdown sections cannot publish a context that would be truncated", async () => {
  const store = new MemoryStore();
  const pipeline = new Pipeline(store, provider);
  const original = await pipeline.ingest(
    scope,
    "file-a",
    "user",
    "original",
    input(),
    signal(),
  );
  await expect(
    pipeline.ingest(
      scope,
      "file-a",
      "user",
      "too-many",
      input("# x\n".repeat(10001)),
      signal(),
    ),
  ).rejects.toMatchObject({ code: "DOCUMENT_CHUNK_LIMIT", status: 413 });
  expect((await store.get(scope, "file-a"))!.generation).toBe(
    original.generation,
  );
  expect(store.events.filter((event) => event === "publish")).toHaveLength(1);
});

test("production embedding requests budget Unicode title and nested headings while preserving source text", async () => {
  const { openAICompatible } = await import("../src/embeddings");
  const requests: string[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      const body = (await request.json()) as { input: string[] };
      requests.push(...body.input);
      if (body.input.some((text) => Buffer.byteLength(text) > 8191))
        return new Response(null, { status: 400 });
      return Response.json({
        data: body.input.map((_, index) => ({
          index,
          embedding: [1, 0, 0, 0, 0, 0, 0, 0],
        })),
      });
    },
  });
  try {
    const adapter = openAICompatible({
      endpoint: `http://127.0.0.1:${server.port}/embeddings`,
      apiKey: "test",
      model: "test",
      dimensions: 8,
    });
    const store = new MemoryStore();
    const text =
      Array.from(
        { length: 6 },
        (_, index) => `${"#".repeat(index + 1)} ${"章".repeat(512)}\n`,
      ).join("") + "文".repeat(1500);
    const document = await new Pipeline(store, adapter).ingest(
      scope,
      "file-a",
      "user",
      "unicode",
      { ...input(text), title: "書".repeat(512) },
      signal(),
    );
    expect(document.chunkCount).toBeGreaterThan(0);
    expect(
      requests.every(
        (text) =>
          text.isWellFormed() &&
          Buffer.byteLength(text) <= adapter.maxInputBytes,
      ),
    ).toBe(true);
    for (const chunk of store.chunks)
      expect(chunk.text).toBe(text.slice(chunk.start, chunk.end));
    expect(store.chunks.at(-1)!.section).toHaveLength(6);
  } finally {
    await server.stop(true);
  }
});
