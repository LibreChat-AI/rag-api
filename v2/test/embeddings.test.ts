import { expect, test } from "bun:test";
import { openAICompatible, QueryEmbeddings } from "../src/embeddings";
import { provider, signal } from "./helpers";

async function serving(
  handler: (request: Request) => Response | Promise<Response>,
  work: (url: string) => Promise<void>,
) {
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: handler });
  try {
    await work(`http://127.0.0.1:${server.port}/embeddings`);
  } finally {
    await server.stop(true);
  }
}
test("provider reconstructs reordered vectors by index and binds dimensions to its space", async () => {
  await serving(
    () =>
      Response.json({
        data: [
          { index: 1, embedding: [0, 1, 0, 0, 0, 0, 0, 0] },
          { index: 0, embedding: [1, 0, 0, 0, 0, 0, 0, 0] },
        ],
      }),
    async (endpoint) => {
      const adapter = openAICompatible({
        endpoint,
        apiKey: "test",
        model: "model",
        dimensions: 8,
      });
      const result = await adapter.embedDocuments(["one", "two"], signal());
      expect(result[0]![0]).toBe(1);
      expect(result[1]![1]).toBe(1);
      expect(adapter.spaceId).not.toBe(
        openAICompatible({
          endpoint,
          apiKey: "test",
          model: "model",
          dimensions: 16,
        }).spaceId,
      );
    },
  );
});
test("malformed vectors, duplicate indices, and redirects fail closed", async () => {
  for (const data of [
    [{ index: 0, embedding: [1] }],
    [
      { index: 0, embedding: Array(8).fill(1) },
      { index: 0, embedding: Array(8).fill(1) },
    ],
  ]) {
    await serving(
      () => Response.json({ data }),
      async (endpoint) => {
        await expect(
          openAICompatible({
            endpoint,
            apiKey: "test",
            model: "model",
            dimensions: 8,
          }).embedDocuments(["one", "two"], signal()),
        ).rejects.toMatchObject({ code: "INVALID_EMBEDDINGS" });
      },
    );
  }
  let requests = 0;
  await serving(
    () => {
      requests++;
      return Response.redirect("http://127.0.0.1:1/secret");
    },
    async (endpoint) => {
      await expect(
        openAICompatible({
          endpoint,
          apiKey: "test",
          model: "model",
          dimensions: 8,
        }).embedQuery("one", signal()),
      ).rejects.toThrow();
    },
  );
  expect(requests).toBeLessThanOrEqual(2);
});
test("query single-flight isolates waiter cancellation and failures do not poison the cache", async () => {
  let count = 0;
  const cached = new QueryEmbeddings({
    ...provider,
    async embedQuery(text, sig) {
      count++;
      await Bun.sleep(20);
      return provider.embedQuery(text, sig);
    },
  });
  const controller = new AbortController();
  const a = cached.get("same query", controller.signal);
  const b = cached.get("same query", signal());
  controller.abort();
  await expect(a).rejects.toThrow();
  await b;
  await cached.get("same query", signal());
  expect(count).toBe(1);
});
