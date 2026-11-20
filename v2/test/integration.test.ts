import { expect, test } from "bun:test";
import {
  ClickHouseStore,
  databaseClient,
  liveWhere,
  vectorSQL,
} from "../src/clickhouse";
import { migrate } from "../src/schema";
import { Pipeline } from "../src/pipeline";
import { ingestSchema, searchSchema } from "../src/contracts";
import { createApp } from "../src/app";
import { auth, provider, signal, token } from "./helpers";

const url = process.env.RAG_V2_TEST_CLICKHOUSE_URL;
for (const quantized of [false, true]) {
  test.skipIf(!url)(
    `real ClickHouse ${quantized ? "quantized" : "plain"}: publication, retries, scope, hybrid, originals, and deletion`,
    async () => {
      const database = `ragv2_test_${crypto.randomUUID().replaceAll("-", "")}`;
      const config = {
        url: url!,
        database: "default",
        username: "default",
        password: "",
      };
      const admin = databaseClient(config);
      let store: ClickHouseStore | undefined;
      try {
        await admin.command({ query: `CREATE DATABASE ${database}` });
        const client = databaseClient({ ...config, database });
        store = new ClickHouseStore(client, database, 64);
        await migrate(client, 64, quantized);
        await store.probe();
        expect(store.quantizedCodesEnabled).toBe(quantized);
        const scope = { tenantId: "tenant-a", namespaceId: "library-a" };
        const pipeline = new Pipeline(store, provider);
        const input = ingestSchema.parse({
          title: "Alpha",
          segments: [
            { kind: "page", index: 1, text: "alpha recovery document" },
          ],
          original: {
            fileId: "file-a",
            revision: "one",
            sha256: "a".repeat(64),
            filename: "original.pdf",
            mediaType: "application/pdf",
          },
        });
        const doc = await pipeline.ingest(
          scope,
          "file-a",
          "user-a",
          "one",
          input,
          signal(),
        );
        const vector = await provider.embedQuery("alpha recovery", signal());
        const search = searchSchema.parse({
          query: "alpha recovery",
          namespaces: [{ namespaceId: "library-a" }],
          precision: quantized ? "quantized" : "exact",
        });
        let hits = await store.search(
          [scope],
          vector,
          provider.spaceId,
          search,
          signal(),
        );
        expect(hits).toHaveLength(1);
        expect(hits[0]!.original).toEqual(input.original);
        expect(hits[0]!.page).toBe(1);
        expect((await store.get(scope, "file-a", signal()))!.version).toBe(
          doc.version,
        );
        expect((await store.context(scope, doc, signal()))[0]!.text).toBe(
          input.segments[0]!.text,
        );
        expect(
          (
            await pipeline.ingest(
              scope,
              "file-a",
              "user-a",
              "one",
              input,
              signal(),
            )
          ).generation,
        ).toBe(doc.generation);
        expect(
          await store.search(
            [{ ...scope, tenantId: "tenant-b" }],
            vector,
            provider.spaceId,
            search,
            signal(),
          ),
        ).toEqual([]);
        expect(
          await store.search(
            [{ ...scope, resourceIds: ["foreign"] }],
            vector,
            provider.spaceId,
            search,
            signal(),
          ),
        ).toEqual([]);
        expect(
          await store.search(
            [scope],
            vector,
            "different-model",
            search,
            signal(),
          ),
        ).toEqual([]);
        // A successfully inserted but unpublished batch must never be searchable.
        const row = {
          tenantId: scope.tenantId,
          namespaceId: scope.namespaceId,
          fileId: "staged",
          generation: "b".repeat(32),
          spaceId: provider.spaceId,
          index: 0,
          text: "alpha recovery staged",
          page: 1,
          segment: 1,
          start: 0,
          end: 21,
          section: [],
          actor: "user:user-a",
          sourceClass: "asserted" as const,
          embedding: vector,
        };
        await store.insert([row], "retry-same-batch", signal());
        await store.insert([row], "retry-same-batch", signal());
        const count = await store.rows<{ n: string }>(
          "SELECT count() AS n FROM chunks WHERE file_id = 'staged'",
          {},
          signal(),
        );
        expect(Number(count[0]!.n)).toBe(1);
        expect(
          (
            await store.search(
              [scope],
              vector,
              provider.spaceId,
              search,
              signal(),
            )
          ).every((hit) => hit.fileId !== "staged"),
        ).toBe(true);
        hits = await store.search(
          [scope],
          vector,
          provider.spaceId,
          { ...search, mode: "hybrid" },
          signal(),
        );
        expect(hits).toHaveLength(1);
        const where = liveWhere([scope], provider.spaceId);
        const plan = await client.query({
          query: `EXPLAIN PLAN ${vectorSQL(where.sql)}`,
          query_params: { ...where.params, vector, k: 5 },
          format: "TabSeparated",
          clickhouse_settings: {
            ...(quantized
              ? {
                  vector_search_use_quantized_codes: "1",
                  vector_search_index_fetch_multiplier: "3",
                }
              : {}),
          },
        });
        const text = await plan.text();
        console.log(
          JSON.stringify({
            qualification: "vector-plan",
            quantized,
            plan: text,
          }),
        );
        expect(text).toContain("ReadFromMergeTree");
        if (quantized) expect(text).toMatch(/Quantized|Rescor|Lazy/i);
        const updated = await pipeline.ingest(
          scope,
          "file-a",
          "user-a",
          "two",
          {
            ...input,
            segments: [
              { kind: "page", index: 1, text: "replacement beta document" },
            ],
          },
          signal(),
        );
        expect(
          (
            await store.search(
              [scope],
              vector,
              provider.spaceId,
              search,
              signal(),
            )
          ).every((hit) => hit.generation === updated.generation),
        ).toBe(true);
        const app = createApp({ auth, store, provider });
        const response = await app.request("/v2/search", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${await token()}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(search),
        });
        expect(response.status).toBe(200);
        await pipeline.delete(scope, "file-a", "user-a", "delete", signal());
        await expect(
          pipeline.ingest(
            scope,
            "file-a",
            "user-a",
            "stale-edit",
            { ...input, ifMatch: updated.generation },
            signal(),
          ),
        ).rejects.toMatchObject({ code: "PRECONDITION_FAILED" });
        expect(
          await store.search(
            [scope],
            vector,
            provider.spaceId,
            search,
            signal(),
          ),
        ).toEqual([]);
        if (!quantized) {
          const { MAX_DOCUMENT_CHUNKS } = await import("../src/contracts");
          const complete = await pipeline.ingest(
            scope,
            "ceiling",
            "user-a",
            "ceiling",
            ingestSchema.parse({
              segments: [
                {
                  kind: "document",
                  index: 1,
                  text: "# x\n".repeat(MAX_DOCUMENT_CHUNKS),
                },
              ],
            }),
            AbortSignal.timeout(30000),
          );
          const context = await store.context(
            scope,
            complete,
            AbortSignal.timeout(10000),
          );
          expect(complete.chunkCount).toBe(MAX_DOCUMENT_CHUNKS);
          expect(context).toHaveLength(MAX_DOCUMENT_CHUNKS);
          expect(context.at(-1)!.index).toBe(MAX_DOCUMENT_CHUNKS - 1);
        }
      } finally {
        await store?.close();
        await admin.command({
          query: `DROP DATABASE IF EXISTS ${database} SYNC`,
        });
        await admin.close();
      }
    },
    60000,
  );
}
