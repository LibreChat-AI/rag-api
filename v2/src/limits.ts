import { RagError } from "./contracts";

export class Gate {
  private active = 0;
  private readonly waiting: Array<() => void> = [];
  constructor(
    private readonly capacity: number,
    private readonly queueSize = 0,
  ) {
    if (!Number.isInteger(capacity) || capacity < 1)
      throw new Error("INVALID_CAPACITY");
  }
  async run<T>(signal: AbortSignal, work: () => Promise<T>): Promise<T> {
    signal.throwIfAborted();
    if (this.active >= this.capacity) {
      if (this.waiting.length >= this.queueSize)
        throw new RagError("OVERLOADED", 503);
      await new Promise<void>((resolve, reject) => {
        const take = () => {
          signal.removeEventListener("abort", cancel);
          resolve();
        };
        const cancel = () => {
          const index = this.waiting.indexOf(take);
          if (index >= 0) this.waiting.splice(index, 1);
          reject(signal.reason);
        };
        this.waiting.push(take);
        signal.addEventListener("abort", cancel, { once: true });
      });
    } else this.active++;
    try {
      signal.throwIfAborted();
      return await work();
    } finally {
      const next = this.waiting.shift();
      if (next) next();
      else this.active--;
    }
  }
}

export class DocumentLocks {
  private readonly active = new Set<string>();
  async run<T>(key: string, work: () => Promise<T>): Promise<T> {
    if (this.active.has(key)) throw new RagError("DOCUMENT_BUSY", 409);
    this.active.add(key);
    try {
      return await work();
    } finally {
      this.active.delete(key);
    }
  }
}

export async function readBounded(
  body: ReadableStream<Uint8Array> | null,
  limit: number,
  signal: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  if (!body) throw new RagError("INVALID_BODY", 400);
  const reader = body.getReader();
  const parts: Uint8Array[] = [];
  let size = 0;
  const cancel = () => {
    void reader.cancel().catch(() => {});
  };
  signal.addEventListener("abort", cancel, { once: true });
  try {
    while (true) {
      signal.throwIfAborted();
      const { value, done } = await reader.read();
      signal.throwIfAborted();
      if (done) break;
      size += value.byteLength;
      if (size > limit) throw new RagError("BODY_LIMIT", 413);
      parts.push(value);
    }
    return Buffer.concat(parts, size);
  } finally {
    signal.removeEventListener("abort", cancel);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
