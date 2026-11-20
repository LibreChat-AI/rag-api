import { expect, test } from "bun:test";
import { scopeWhere, liveWhere, vectorSQL } from "../src/clickhouse";

test("library/file predicates preserve correlated grants instead of cross-product access", () => {
  const where = scopeWhere([
    { tenantId: "tenant-a", namespaceId: "a", resourceIds: ["one"] },
    { tenantId: "tenant-a", namespaceId: "b", resourceIds: ["two"] },
  ]);
  expect(where.sql).toContain(
    "namespace_id = {lib0:String} AND file_id IN {files0:Array(String)}",
  );
  expect(where.sql).toContain(
    "namespace_id = {lib1:String} AND file_id IN {files1:Array(String)}",
  );
  expect(where.params).toEqual({
    tenant: "tenant-a",
    lib0: "a",
    files0: ["one"],
    lib1: "b",
    files1: ["two"],
  });
  expect(() =>
    scopeWhere([
      { tenantId: "a", namespaceId: "x" },
      { tenantId: "b", namespaceId: "x" },
    ]),
  ).toThrow();
});
test("publication is a scoped semi-join filter; the outer vector scan stays eligible", () => {
  const where = liveWhere(
    [{ tenantId: "a", namespaceId: "x", resourceIds: ["one"] }],
    "space",
  );
  const sql = vectorSQL(where.sql);
  expect(sql).toContain("FROM documents FINAL");
  expect(sql).toContain("state = 'ready'");
  expect(sql).toContain(
    "cosineDistance(embedding, {vector:Array(Float32)}) AS distance",
  );
  expect(sql).toContain("ORDER BY distance ASC LIMIT {k:UInt32}");
  expect(sql).not.toContain("FROM chunks FINAL");
  expect(sql).not.toContain(" JOIN ");
  expect(sql).not.toContain("LIMIT 1 BY");
});

test("context never silently returns a clipped, missing, or duplicate chunk sequence", async () => {
  const { ClickHouseStore, databaseClient } = await import("../src/clickhouse");
  const { MAX_DOCUMENT_CHUNKS, ingestSchema } =
    await import("../src/contracts");
  const { Pipeline } = await import("../src/pipeline");
  const { MemoryStore, provider, signal } = await import("./helpers");
  const scope = { tenantId: "tenant-a", namespaceId: "library-a" };
  const document = await new Pipeline(new MemoryStore(), provider).ingest(
    scope,
    "file-a",
    "user",
    "key",
    ingestSchema.parse({
      segments: [{ kind: "document", index: 1, text: "alpha" }],
    }),
    signal(),
  );
  const client = databaseClient({
    url: "http://127.0.0.1:1",
    database: "test",
    username: "default",
    password: "",
  });
  const store = new ClickHouseStore(client, "test", 64);
  const row = {
    chunk_index: 0,
    content: "alpha",
    page: null,
    segment: 1,
    char_start: 0,
    char_end: 5,
    section: [],
  };
  let rows = [row];
  let calls = 0;
  store.rows = async <T>(query: string, params: Record<string, unknown>) => {
    calls++;
    expect(query).toContain("LIMIT {limit:UInt32}");
    expect(params.limit).toBe(MAX_DOCUMENT_CHUNKS + 1);
    return rows as T[];
  };
  try {
    expect(await store.context(scope, document, signal())).toHaveLength(1);
    for (const invalid of [[], [row, row], [{ ...row, chunk_index: 1 }]]) {
      rows = invalid;
      await expect(
        store.context(scope, document, signal()),
      ).rejects.toMatchObject({ code: "CONTEXT_INCOMPLETE" });
    }
    const before = calls;
    await expect(
      store.context(
        scope,
        { ...document, chunkCount: MAX_DOCUMENT_CHUNKS + 1 },
        signal(),
      ),
    ).rejects.toMatchObject({ code: "DOCUMENT_CHUNK_LIMIT" });
    expect(calls).toBe(before);
  } finally {
    await store.close();
  }
});
