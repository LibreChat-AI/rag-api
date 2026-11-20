import type { ClickHouseClient } from "@clickhouse/client";

export async function migrate(
  client: ClickHouseClient,
  dimensions: number,
  quantized = false,
): Promise<void> {
  if (
    !Number.isInteger(dimensions) ||
    dimensions < 8 ||
    dimensions > 8192 ||
    (quantized && dimensions % 8)
  )
    throw new Error("INVALID_DIMENSIONS");
  const statements = [
    `CREATE TABLE IF NOT EXISTS documents (
      tenant_id LowCardinality(String), namespace_id LowCardinality(String), file_id String,
      generation FixedString(32), version UInt64, state Enum8('ready'=1,'deleted'=2), title String,
      original String, actor String, source_class Enum8('asserted'=1,'extracted'=2), space_id LowCardinality(String), chunk_count UInt32, operation_key FixedString(64), request_hash FixedString(64)
    ) ENGINE = ReplacingMergeTree(version) ORDER BY (tenant_id, namespace_id, file_id) SETTINGS index_granularity=64, non_replicated_deduplication_window=10000`,
    `CREATE TABLE IF NOT EXISTS chunks (
      tenant_id LowCardinality(String), namespace_id LowCardinality(String), space_id LowCardinality(String),
      file_id String, generation FixedString(32), chunk_index UInt32, content String,
      content_lc String MATERIALIZED lowerUTF8(content), page Nullable(UInt16), segment UInt16,
      char_start UInt32, char_end UInt32, section Array(String), actor String, source_class Enum8('asserted'=1,'extracted'=2),
      embedding Array(${quantized ? "BFloat16" : "Float32"}) ${quantized ? `CODEC(Quantized('rabitq', ${dimensions}, 0))` : "CODEC(NONE)"},
      CONSTRAINT vector_width CHECK length(embedding) = ${dimensions},
      INDEX content_text content_lc TYPE text(tokenizer='splitByNonAlpha') GRANULARITY 100000000
    ) ENGINE=MergeTree PARTITION BY cityHash64(tenant_id, namespace_id)%16
      ORDER BY (tenant_id, namespace_id, space_id, file_id, generation, chunk_index)
      SETTINGS non_replicated_deduplication_window=10000, exclude_materialize_skip_indexes_on_merge='content_text'`,
    `CREATE TABLE IF NOT EXISTS operations (
      tenant_id LowCardinality(String), namespace_id LowCardinality(String), operation_key FixedString(64), receipt String, version UInt64
    ) ENGINE=ReplacingMergeTree(version) ORDER BY (tenant_id, namespace_id, operation_key)
      SETTINGS index_granularity=64, non_replicated_deduplication_window=10000`,
  ];
  for (const query of statements)
    await client.command({
      query,
      clickhouse_settings: {
        log_comment: "ragv2.migrate",
        ...(quantized ? { enable_quantized_codec: "1" } : {}),
      },
    });
}
