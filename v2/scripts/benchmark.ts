import type { ClickHouseSettings } from "@clickhouse/client";
import {
  databaseClient,
  ClickHouseStore,
  liveWhere,
  vectorSQL,
} from "../src/clickhouse";
import { migrate } from "../src/schema";
import { createApp } from "../src/app";
import { auth, token } from "../test/helpers";
import type { Document, StoredChunk } from "../src/contracts";
import type { EmbeddingProvider } from "../src/embeddings";

const url = process.env.RAG_V2_TEST_CLICKHOUSE_URL;
if (!url) throw Error("RAG_V2_TEST_CLICKHOUSE_URL_REQUIRED");
const dimensions = Number(process.env.RAG_V2_BENCH_DIMENSIONS ?? 1536);
const count = Number(process.env.RAG_V2_BENCH_ROWS ?? 10000);
if (!Number.isInteger(count) || count < 100 || count > 100000)
  throw Error("INVALID_BENCH_ROWS");
const database = `ragv2_bench_${crypto.randomUUID().replaceAll("-", "")}`;
const config = { url, database: "default", username: "default", password: "" };
const admin = databaseClient(config);
const client = databaseClient({ ...config, database });
const store = new ClickHouseStore(client, database, dimensions);
const signal = () => AbortSignal.timeout(30000);
let seed = 17;
const vector = () =>
  Array.from({ length: dimensions }, () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296 - 0.5;
  });
const reference = vector();
const scope = { tenantId: "benchmark", namespaceId: "library-a" };
const generation = "a".repeat(32);
const spaceId = `synthetic-${dimensions}`;
const provider: EmbeddingProvider = {
  spaceId,
  dimensions,
  maxInputBytes: 8191,
  async embedQuery() {
    return reference;
  },
  async embedDocuments(texts) {
    return texts.map(vector);
  },
};
let server: ReturnType<typeof Bun.serve> | undefined;
const percentile = (values: number[], fraction: number) => {
  const sorted = [...values].sort((a, b) => a - b);
  return Number(
    sorted[
      Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))
    ]!.toFixed(2),
  );
};
try {
  await admin.command({ query: `CREATE DATABASE ${database}` });
  await migrate(client, dimensions, true);
  await store.probe();
  await client.command({
    query:
      "CREATE TABLE retired (tenant_id String, namespace_id String, space_id String, file_id String, generation FixedString(32)) ENGINE=MergeTree ORDER BY (tenant_id,namespace_id,space_id,file_id,generation)",
  });
  const start = performance.now();
  for (let offset = 0; offset < count; offset += 64) {
    const rows: StoredChunk[] = Array.from(
      { length: Math.min(64, count - offset) },
      (_, i) => ({
        ...scope,
        fileId: "corpus",
        generation,
        spaceId,
        index: offset + i,
        text: `synthetic chunk ${offset + i}`,
        page: null,
        segment: 1,
        start: 0,
        end: 20,
        section: [],
        actor: "service:benchmark",
        sourceClass: "asserted",
        embedding: vector(),
      }),
    );
    await store.insert(rows, `bench-${offset}`, signal());
  }
  const document: Document = {
    ...scope,
    fileId: "corpus",
    generation,
    spaceId,
    version: "1",
    state: "ready",
    title: "Corpus",
    original: null,
    actor: "service:benchmark",
    sourceClass: "asserted",
    chunkCount: count,
    operationKey: "a".repeat(64),
    requestHash: "a".repeat(64),
  };
  await store.publish(document, signal());
  const insertedMs = performance.now() - start;
  const where = liveWhere([scope], spaceId);
  const baselineWhere =
    "tenant_id = {tenant:String} AND namespace_id = {lib0:String} AND space_id = {space:String} AND (tenant_id,namespace_id,space_id,file_id,generation) NOT IN (SELECT tenant_id,namespace_id,space_id,file_id,generation FROM retired)";
  const params = { ...where.params, vector: reference, k: 10 };
  const settings: ClickHouseSettings = {
    vector_search_use_quantized_codes: "1",
    vector_search_index_fetch_multiplier: "3",
    use_query_condition_cache: 0,
    use_query_cache: 0,
    max_threads: 2,
  };
  const times = {
    baseline: [] as number[],
    publication: [] as number[],
    httpExact: [] as number[],
    httpQuantized: [] as number[],
  };
  for (let round = 0; round < 24; round++) {
    for (const mode of (round % 2
      ? ["publication", "baseline"]
      : ["baseline", "publication"]) as Array<"baseline" | "publication">) {
      const started = performance.now();
      const result = await client.query({
        query: vectorSQL(mode === "baseline" ? baselineWhere : where.sql),
        query_params: params,
        format: "JSONEachRow",
        clickhouse_settings: {
          ...settings,
          log_comment: `ragv2.benchmark.${mode}`,
        },
      });
      await result.json();
      if (round >= 4) times[mode].push(performance.now() - started);
    }
  }
  const exact = await client.query({
    query: vectorSQL(where.sql),
    query_params: params,
    format: "JSONEachRow",
    clickhouse_settings: {
      vector_search_use_quantized_codes: "0",
      use_query_condition_cache: 0,
    },
  });
  const nearest = await exact.json<{ chunk_index: number }>();
  const app = createApp({ auth, store, provider });
  server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: app.fetch });
  const jwt = await token({
    tenant_id: "benchmark",
    grants: [
      {
        namespaceId: "library-a",
        resourceKind: "document",
        operations: ["read"],
      },
    ],
  });
  const recall = { exact: 0, quantized: 0 };
  for (let round = 0; round < 24; round++) {
    for (const precision of ["exact", "quantized"] as const) {
      const started = performance.now();
      const response = await fetch(
        `http://127.0.0.1:${server.port}/v2/search`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${jwt}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: "benchmark query",
            namespaces: [{ namespaceId: "library-a" }],
            k: 10,
            precision,
          }),
        },
      );
      if (!response.ok) throw Error(`BENCHMARK_HTTP_${response.status}`);
      const body = (await response.json()) as {
        hits: Array<{ index: number }>;
      };
      recall[precision] =
        body.hits.filter((hit) =>
          nearest.some((exact) => exact.chunk_index === hit.index),
        ).length / 10;
      if (round >= 4)
        times[precision === "exact" ? "httpExact" : "httpQuantized"].push(
          performance.now() - started,
        );
    }
  }
  console.log(
    JSON.stringify({
      benchmark: "synthetic-single-namespace-warm-cache",
      runtime: Bun.version,
      dimensions,
      rows: count,
      insertMs: Number(insertedMs.toFixed(2)),
      recallAt10: recall,
      arms: Object.fromEntries(
        Object.entries(times).map(([mode, values]) => [
          mode,
          { p50Ms: percentile(values, 0.5), p95Ms: percentile(values, 0.95) },
        ]),
      ),
      note: "Baseline is Loom-shaped SQL over identical coded data, not a running Loom deployment. HTTP includes JWT verification, vector cache, candidate search, and metadata hydration. No external provider or extraction latency.",
    }),
  );
} finally {
  await server?.stop(true);
  await store.close();
  await admin.command({ query: `DROP DATABASE IF EXISTS ${database} SYNC` });
  await admin.close();
}
