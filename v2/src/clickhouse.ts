import {
  createClient,
  TupleParam,
  ClickHouseLogLevel,
  type ClickHouseClient,
  type ClickHouseSettings,
} from "@clickhouse/client";
import { AsyncLocalStorage } from "node:async_hooks";
import { createHash } from "node:crypto";
import { Gate } from "./limits";
import {
  RagError,
  narrowScope,
  MAX_DOCUMENT_CHUNKS,
  type Chunk,
  type Document,
  type Hit,
  type Receipt,
  type Scope,
  type SearchInput,
  type Store,
  type StoredChunk,
} from "./contracts";

export const queryTag = new AsyncLocalStorage<string>();
export type DatabaseConfig = {
  url: string;
  username: string;
  password: string;
  database: string;
};
export function databaseClient(config: DatabaseConfig): ClickHouseClient {
  if (!/^[a-z][a-z0-9_]{0,62}$/.test(config.database))
    throw new Error("INVALID_DATABASE");
  return createClient({
    ...config,
    application: "rag-api-v2-poc",
    log: { level: ClickHouseLogLevel.OFF },
    max_open_connections: 12,
    request_timeout: 15000,
    keep_alive: { enabled: true, idle_socket_ttl: 2500 },
    compression: { request: true, response: true },
    clickhouse_settings: {
      async_insert: 0,
      wait_for_async_insert: 1,
      output_format_json_quote_64bit_integers: 1,
      max_execution_time: 10,
      enable_filesystem_cache: 1,
    },
  });
}

export function scopeWhere(scopes: readonly Scope[]): {
  sql: string;
  params: Record<string, unknown>;
} {
  if (
    !scopes.length ||
    scopes.some((scope) => scope.tenantId !== scopes[0]!.tenantId)
  )
    throw new RagError("SCOPE_DENIED", 403);
  const params: Record<string, unknown> = { tenant: scopes[0]!.tenantId };
  const clauses = scopes.map((scope, index) => {
    params[`lib${index}`] = scope.namespaceId;
    if (scope.resourceIds !== undefined)
      params[`files${index}`] = scope.resourceIds;
    return `(namespace_id = {lib${index}:String}${scope.resourceIds !== undefined ? ` AND file_id IN {files${index}:Array(String)}` : ""})`;
  });
  return {
    sql: `tenant_id = {tenant:String} AND (${clauses.join(" OR ")})`,
    params,
  };
}
export function liveWhere(
  scopes: readonly Scope[],
  spaceId: string,
): { sql: string; params: Record<string, unknown> } {
  const scope = scopeWhere(scopes);
  return {
    sql: `${scope.sql} AND space_id = {space:String} AND (tenant_id, namespace_id, file_id, generation) IN (SELECT tenant_id, namespace_id, file_id, generation FROM documents FINAL WHERE ${scope.sql} AND state = 'ready' AND space_id = {space:String})`,
    params: { ...scope.params, space: spaceId },
  };
}
export function vectorSQL(where: string): string {
  return `SELECT namespace_id, file_id, generation, chunk_index, content, page, segment, char_start, char_end, section,
    cosineDistance(embedding, {vector:Array(Float32)}) AS distance FROM chunks
    WHERE ${where} ORDER BY distance ASC LIMIT {k:UInt32}`;
}
type DocumentRow = {
  tenant_id: string;
  namespace_id: string;
  file_id: string;
  generation: string;
  version: string;
  state: "ready" | "deleted";
  title: string;
  original: string;
  actor: string;
  source_class: "asserted" | "extracted";
  space_id: string;
  chunk_count: number;
  operation_key: string;
  request_hash: string;
};
function hydrate(row: DocumentRow): Document {
  return {
    tenantId: row.tenant_id,
    namespaceId: row.namespace_id,
    fileId: row.file_id,
    generation: row.generation,
    version: row.version,
    state: row.state,
    title: row.title,
    original: JSON.parse(row.original),
    actor: row.actor,
    sourceClass: row.source_class,
    spaceId: row.space_id,
    chunkCount: row.chunk_count,
    operationKey: row.operation_key,
    requestHash: row.request_hash,
  };
}
const documentColumns =
  "tenant_id, namespace_id, file_id, generation, toString(version) AS version, state, title, original, actor, source_class, space_id, chunk_count, operation_key, request_hash";
type HitRow = {
  namespace_id: string;
  file_id: string;
  generation: string;
  chunk_index: number;
  content: string;
  page: number | null;
  segment: number;
  char_start: number;
  char_end: number;
  section: string[];
  distance: number;
  lexical?: number;
};
const transient = (error: unknown) =>
  error instanceof Error &&
  /ECONNRESET|ECONNREFUSED|EPIPE|socket hang up/.test(error.message);

export class ClickHouseStore implements Store {
  private readonly reads = new Gate(8, 8);
  private readonly writes = new Gate(2, 4);
  private quantized = false;
  private codesSettingSupported = false;
  private probeAt = 0;
  private probing: Promise<void> | undefined;
  get quantizedCodesEnabled(): boolean {
    return this.quantized;
  }
  private readonly stats = new Map<
    string,
    { expires: number; n: number; avgdl: number; df: number[] }
  >();
  constructor(
    public readonly client: ClickHouseClient,
    private readonly database: string,
    public readonly dimensions: number,
  ) {}
  async probe(): Promise<void> {
    this.probing ??= this.probeState().finally(() => {
      this.probing = undefined;
    });
    return this.probing;
  }
  private async probeState(): Promise<void> {
    const signal = AbortSignal.timeout(5000);
    try {
      const rows = await this.rows<{
        supported: number;
        coded: number;
        patches: number;
      }>(
        `SELECT (SELECT count() FROM system.settings WHERE name = 'vector_search_use_quantized_codes') AS supported,
         (SELECT count() FROM system.columns WHERE database = {db:String} AND table = 'chunks' AND name = 'embedding' AND compression_codec LIKE '%Quantized(%') AS coded,
         (SELECT count() FROM system.parts WHERE database = {db:String} AND table = 'chunks' AND active AND startsWith(name, 'patch')) AS patches`,
        { db: this.database },
        signal,
      );
      this.codesSettingSupported = Number(rows[0]?.supported) > 0;
      this.quantized =
        Number(rows[0]?.supported) > 0 &&
        Number(rows[0]?.coded) > 0 &&
        Number(rows[0]?.patches) === 0;
      if (this.quantized) {
        try {
          const replicas = await this.rows<{
            replicas: string;
            supported: string;
          }>(
            "SELECT count() AS replicas, countIf(has_setting) AS supported FROM (SELECT max(name = 'vector_search_use_quantized_codes') AS has_setting FROM clusterAllReplicas('default', system.settings) GROUP BY hostName())",
            {},
            signal,
          );
          this.quantized =
            Number(replicas[0]?.replicas) > 0 &&
            replicas[0]?.supported === replicas[0]?.replicas;
        } catch {
          /* Standalone OSS has no default cluster. Reads still handle unknown settings. */
        }
      }
    } catch {
      this.quantized = false;
      this.codesSettingSupported = false;
    }
    this.probeAt = Date.now();
  }
  private settings(extra: ClickHouseSettings = {}): ClickHouseSettings {
    return {
      log_comment: (queryTag.getStore() ?? "ragv2.background").slice(0, 120),
      ...extra,
    };
  }
  async rows<T>(
    query: string,
    params: Record<string, unknown>,
    signal: AbortSignal,
    extra: ClickHouseSettings = {},
  ): Promise<T[]> {
    return this.reads.run(signal, async () => {
      for (let attempt = 0; ; attempt++) {
        try {
          const result = await this.client.query({
            query,
            query_params: params,
            format: "JSONEachRow",
            abort_signal: signal,
            clickhouse_settings: this.settings(extra),
          });
          try {
            return await result.json<T>();
          } finally {
            result.close();
          }
        } catch (error) {
          signal.throwIfAborted();
          if (attempt || !transient(error)) throw error;
        }
      }
    });
  }
  private async write(
    table: string,
    values: readonly Record<string, unknown>[],
    token: string,
    signal: AbortSignal,
  ): Promise<void> {
    await this.writes.run(signal, async () => {
      for (let attempt = 0; ; attempt++) {
        try {
          await this.client.insert({
            table,
            values,
            format: "JSONEachRow",
            abort_signal: signal,
            clickhouse_settings: this.settings({
              async_insert: 0,
              wait_for_async_insert: 1,
              insert_deduplicate: 1,
              insert_deduplication_token: token,
            }),
          });
          return;
        } catch (error) {
          signal.throwIfAborted();
          if (attempt || !transient(error)) throw error;
        }
      }
    });
  }
  async get(
    scope: Scope,
    fileId: string,
    signal: AbortSignal,
  ): Promise<Document | null> {
    const where = scopeWhere([narrowScope(scope, fileId)]);
    const rows = await this.rows<DocumentRow>(
      `SELECT ${documentColumns} FROM documents FINAL WHERE ${where.sql} LIMIT 1`,
      where.params,
      signal,
      { select_sequential_consistency: "1" },
    );
    return rows[0] ? hydrate(rows[0]) : null;
  }
  async receipt(
    scope: Scope,
    operationKey: string,
    signal: AbortSignal,
  ): Promise<Receipt | null> {
    const rows = await this.rows<{ receipt: string }>(
      "SELECT receipt FROM operations FINAL WHERE tenant_id = {tenant:String} AND namespace_id = {library:String} AND operation_key = {key:String} LIMIT 1",
      { tenant: scope.tenantId, library: scope.namespaceId, key: operationKey },
      signal,
      { select_sequential_consistency: "1" },
    );
    return rows[0] ? JSON.parse(rows[0].receipt) : null;
  }
  async insert(
    chunks: readonly StoredChunk[],
    batchId: string,
    signal: AbortSignal,
  ): Promise<void> {
    await this.write(
      "chunks",
      chunks.map((chunk) => ({
        tenant_id: chunk.tenantId,
        namespace_id: chunk.namespaceId,
        file_id: chunk.fileId,
        generation: chunk.generation,
        space_id: chunk.spaceId,
        chunk_index: chunk.index,
        content: chunk.text,
        page: chunk.page,
        segment: chunk.segment,
        char_start: chunk.start,
        char_end: chunk.end,
        section: chunk.section,
        actor: chunk.actor,
        source_class: chunk.sourceClass,
        embedding: chunk.embedding,
      })),
      batchId,
      signal,
    );
  }
  async publish(document: Document, signal: AbortSignal): Promise<void> {
    await this.write(
      "documents",
      [
        {
          tenant_id: document.tenantId,
          namespace_id: document.namespaceId,
          file_id: document.fileId,
          generation: document.generation,
          version: document.version,
          state: document.state,
          title: document.title,
          original: JSON.stringify(document.original),
          actor: document.actor,
          source_class: document.sourceClass,
          space_id: document.spaceId,
          chunk_count: document.chunkCount,
          operation_key: document.operationKey,
          request_hash: document.requestHash,
        },
      ],
      `${document.operationKey}:${document.version}`,
      signal,
    );
    this.stats.clear();
  }
  async record(
    scope: Scope,
    operationKey: string,
    receipt: Receipt,
    signal: AbortSignal,
  ): Promise<void> {
    await this.write(
      "operations",
      [
        {
          tenant_id: scope.tenantId,
          namespace_id: scope.namespaceId,
          operation_key: operationKey,
          receipt: JSON.stringify(receipt),
          version: receipt.document.version,
        },
      ],
      `${operationKey}:${receipt.document.version}`,
      signal,
    );
  }
  async search(
    scopes: readonly Scope[],
    vector: readonly number[],
    spaceId: string,
    input: SearchInput,
    signal: AbortSignal,
  ): Promise<Hit[]> {
    if (Date.now() - this.probeAt > 60000) await this.probe();
    const where = liveWhere(scopes, spaceId);
    const tokens = [
      ...new Set(input.query.toLowerCase().match(/[a-z0-9]{2,}/g) ?? []),
    ].slice(0, 16);
    const k = input.mode === "hybrid" ? Math.max(input.k * 3, 20) : input.k;
    const params = { ...where.params, vector, k };
    const vectorQuery = async () => {
      try {
        return await this.rows<HitRow>(vectorSQL(where.sql), params, signal, {
          select_sequential_consistency: "1",
          use_query_condition_cache: 0,
          ...(this.quantized && input.precision === "quantized"
            ? {
                vector_search_use_quantized_codes: "1",
                vector_search_index_fetch_multiplier: "3",
              }
            : this.codesSettingSupported
              ? { vector_search_use_quantized_codes: "0" }
              : {}),
        });
      } catch (error) {
        if (
          !this.codesSettingSupported ||
          !(error instanceof Error) ||
          !(
            ("code" in error && String(error.code) === "115") ||
            /Unknown setting|UNKNOWN_SETTING/i.test(error.message)
          )
        )
          throw error;
        this.quantized = false;
        this.codesSettingSupported = false;
        return this.rows<HitRow>(vectorSQL(where.sql), params, signal, {
          select_sequential_consistency: "1",
          use_query_condition_cache: 0,
        });
      }
    };
    const [semantic, lexical] = await Promise.all([
      vectorQuery(),
      input.mode === "hybrid" && tokens.length
        ? this.lexical(where, params, tokens, signal)
        : Promise.resolve([]),
    ]);
    const byKey = new Map<string, { row: HitRow; score: number }>();
    const keyOf = (row: HitRow) =>
      JSON.stringify([
        row.namespace_id,
        row.file_id,
        row.generation,
        row.chunk_index,
      ]);
    for (const [index, row] of semantic.entries())
      byKey.set(keyOf(row), {
        row,
        score: lexical.length ? 1 / (60 + index + 1) : 1 - row.distance,
      });
    for (const [index, row] of lexical.entries()) {
      const key = keyOf(row);
      const prior = byKey.get(key);
      byKey.set(key, {
        row: prior?.row ?? row,
        score: (prior?.score ?? 0) + 1 / (60 + index + 1),
      });
    }
    const top = [...byKey.values()]
      .sort((a, b) => b.score - a.score || a.row.distance - b.row.distance)
      .slice(0, input.k);
    if (!top.length) return [];
    const scoped = scopeWhere(scopes);
    const docs = await this.rows<DocumentRow>(
      `SELECT ${documentColumns} FROM documents FINAL WHERE ${scoped.sql} AND state = 'ready' AND (namespace_id, file_id, generation) IN {keys:Array(Tuple(String, String, String))}`,
      {
        ...scoped.params,
        keys: top.map(
          ({ row }) =>
            new TupleParam([row.namespace_id, row.file_id, row.generation]),
        ),
      },
      signal,
      { select_sequential_consistency: "1" },
    );
    const metadata = new Map(
      docs.map((row) => [
        JSON.stringify([row.namespace_id, row.file_id, row.generation]),
        hydrate(row),
      ]),
    );
    return top.flatMap(({ row, score }) => {
      const doc = metadata.get(
        JSON.stringify([row.namespace_id, row.file_id, row.generation]),
      );
      return doc
        ? [
            {
              namespaceId: row.namespace_id,
              fileId: row.file_id,
              generation: row.generation,
              index: row.chunk_index,
              text: row.content,
              page: row.page,
              segment: row.segment,
              start: row.char_start,
              end: row.char_end,
              section: row.section,
              distance: row.distance,
              score,
              original: doc.original,
              actor: doc.actor,
              sourceClass: doc.sourceClass,
            },
          ]
        : [];
    });
  }
  private async lexical(
    where: ReturnType<typeof liveWhere>,
    params: Record<string, unknown>,
    tokens: string[],
    signal: AbortSignal,
  ): Promise<HitRow[]> {
    const key = createHash("sha256")
      .update(JSON.stringify([where.params, tokens]))
      .digest("hex");
    let stats = this.stats.get(key);
    if (!stats || stats.expires <= Date.now()) {
      const tokenParams = Object.fromEntries(
        tokens.map((token, index) => [`term${index}`, token]),
      );
      const dfSQL = tokens
        .map((_, index) => `countIf(has(words, {term${index}:String}))`)
        .join(", ");
      const rows = await this.rows<{ n: string; avgdl: number; df: string[] }>(
        `WITH splitByNonAlpha(lowerUTF8(content)) AS words SELECT count() AS n, avg(length(words)) AS avgdl, [${dfSQL}] AS df FROM chunks WHERE ${where.sql}`,
        { ...params, ...tokenParams },
        signal,
        { select_sequential_consistency: "1" },
      );
      const row = rows[0];
      if (!row || !Number(row.n) || !row.avgdl) return [];
      stats = {
        n: Number(row.n),
        avgdl: row.avgdl,
        df: row.df.map(Number),
        expires: Date.now() + 30000,
      };
      this.stats.set(key, stats);
      while (this.stats.size > 128)
        this.stats.delete(this.stats.keys().next().value!);
    }
    const idfs = stats.df.map((df) =>
      Math.log(1 + (stats.n - df + 0.5) / (df + 0.5)),
    );
    return this.rows<HitRow>(
      `WITH splitByNonAlpha(content_lc) AS words SELECT namespace_id, file_id, generation, chunk_index, content, page, segment, char_start, char_end, section, cosineDistance(embedding, {vector:Array(Float32)}) AS distance,
      arraySum(arrayMap((tok, idf) -> idf * countEqual(words, tok) * 2.2 / (countEqual(words, tok) + 1.2 * (0.25 + 0.75 * length(words) / {avgdl:Float32})), {tokens:Array(String)}, {idfs:Array(Float32)})) AS lexical
      FROM chunks WHERE ${where.sql} AND hasAnyTokens(content_lc, {tokens:Array(String)}) ORDER BY lexical DESC, distance ASC LIMIT {k:UInt32}`,
      { ...params, tokens, idfs, avgdl: stats.avgdl },
      signal,
      { select_sequential_consistency: "1" },
    );
  }
  async context(
    scope: Scope,
    document: Document,
    signal: AbortSignal,
  ): Promise<Chunk[]> {
    if (document.chunkCount > MAX_DOCUMENT_CHUNKS)
      throw new RagError("DOCUMENT_CHUNK_LIMIT", 413);
    const where = liveWhere(
      [narrowScope(scope, document.fileId)],
      document.spaceId,
    );
    const rows = await this.rows<HitRow>(
      `SELECT chunk_index, content, page, segment, char_start, char_end, section FROM chunks WHERE ${where.sql} AND generation = {generation:String} ORDER BY chunk_index ASC LIMIT {limit:UInt32}`,
      {
        ...where.params,
        generation: document.generation,
        limit: MAX_DOCUMENT_CHUNKS + 1,
      },
      signal,
      { select_sequential_consistency: "1" },
    );
    if (
      rows.length !== document.chunkCount ||
      rows.some((row, index) => row.chunk_index !== index)
    ) {
      throw new RagError("CONTEXT_INCOMPLETE", 503);
    }
    return rows.map((row) => ({
      index: row.chunk_index,
      text: row.content,
      page: row.page,
      segment: row.segment,
      start: row.char_start,
      end: row.char_end,
      section: row.section,
    }));
  }
  close(): Promise<void> {
    return this.client.close();
  }
}
