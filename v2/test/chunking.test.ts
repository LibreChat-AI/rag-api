import { expect, test } from "bun:test";
import { chunks } from "../src/chunking";
import { ingestSchema } from "../src/contracts";
import { Gate, readBounded } from "../src/limits";
import { signal } from "./helpers";

test("chunks never cross page boundaries and preserve UTF-16 source offsets and section breadcrumbs", () => {
  const segments = [
    {
      kind: "page" as const,
      index: 1,
      text: "# Heading\n" + "😀word ".repeat(500),
    },
    { kind: "page" as const, index: 2, text: "Second page" },
  ];
  const result = [...chunks(segments)];
  expect(result.map((chunk) => chunk.index)).toEqual(
    result.map((_, index) => index),
  );
  for (const chunk of result) {
    expect(chunk.text).toBe(
      segments[chunk.page! - 1]!.text.slice(chunk.start, chunk.end),
    );
    expect(chunk.text.isWellFormed()).toBe(true);
  }
  expect(result[0]!.section).toEqual(["Heading"]);
  expect(result.at(-1)!.page).toBe(2);
});
test("rejects unordered, oversized and empty extraction output", () => {
  for (const segments of [
    [{ kind: "page", index: 2, text: "text" }],
    [{ kind: "document", index: 1, text: " " }],
    [{ kind: "document", index: 1, text: "x".repeat(1024 * 1024 + 1) }],
  ])
    expect(() => ingestSchema.parse({ segments })).toThrow();
});
test("admission is bounded, queued cancellation frees its position, and slots recover after failure", async () => {
  const gate = new Gate(1, 1);
  let release: () => void = () => {};
  const active = gate.run(
    signal(),
    () =>
      new Promise<void>((resolve) => {
        release = resolve;
      }),
  );
  const controller = new AbortController();
  const queued = gate.run(controller.signal, async () => {});
  await expect(gate.run(signal(), async () => {})).rejects.toMatchObject({
    code: "OVERLOADED",
  });
  controller.abort();
  await expect(queued).rejects.toThrow();
  release();
  await active;
  await expect(
    gate.run(signal(), async () => {
      throw Error("work");
    }),
  ).rejects.toThrow();
  expect(await gate.run(signal(), async () => 1)).toBe(1);
});
test("body cap rejects without reading the rest of an untrusted stream", async () => {
  let cancelled = false;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new Uint8Array(20));
    },
    cancel() {
      cancelled = true;
    },
  });
  await expect(readBounded(body, 10, signal())).rejects.toMatchObject({
    code: "BODY_LIMIT",
  });
  expect(cancelled).toBe(true);
});

test("the shared chunk ceiling accepts exactly 10,000 sections and rejects the next", async () => {
  const { MAX_DOCUMENT_CHUNKS } = await import("../src/contracts");
  const sections = (count: number) => [
    { kind: "document" as const, index: 1, text: "# x\n".repeat(count) },
  ];
  const accepted = [...chunks(sections(MAX_DOCUMENT_CHUNKS))];
  expect(accepted).toHaveLength(MAX_DOCUMENT_CHUNKS);
  expect(accepted.at(-1)!.index).toBe(MAX_DOCUMENT_CHUNKS - 1);
  expect(() => [...chunks(sections(MAX_DOCUMENT_CHUNKS + 1))]).toThrow(
    "DOCUMENT_CHUNK_LIMIT",
  );
});

test("embedding prefixes end on UTF-8 boundaries without dropping chunk content", async () => {
  const { documentEmbeddingInput } = await import("../src/chunking");
  const chunk = {
    index: 0,
    text: "正文😀",
    page: null,
    segment: 1,
    start: 0,
    end: 4,
    section: ["章节😀"],
  };
  for (let limit = Buffer.byteLength(chunk.text); limit < 40; limit++) {
    const embedded = documentEmbeddingInput(chunk, "标题😀", limit);
    expect(Buffer.byteLength(embedded)).toBeLessThanOrEqual(limit);
    expect(embedded.endsWith(chunk.text)).toBe(true);
    expect(embedded.isWellFormed()).toBe(true);
  }
  expect(() => documentEmbeddingInput(chunk, "", 1)).toThrow(
    "EMBEDDING_INPUT_LIMIT",
  );
});
