import { z } from "zod";
import {
  deterministicProvider,
  openAICompatible,
  type EmbeddingProvider,
} from "./embeddings";
import type { DatabaseConfig } from "./clickhouse";

export function databaseConfig(): DatabaseConfig {
  const url = process.env.RAG_V2_CLICKHOUSE_URL ?? "http://localhost:8123";
  const parsed = new URL(url);
  if (
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash ||
    (parsed.protocol !== "https:" &&
      !(
        parsed.protocol === "http:" &&
        (["localhost", "127.0.0.1", "[::1]"].includes(parsed.hostname) ||
          process.env.RAG_V2_ALLOW_INSECURE_CLICKHOUSE === "true")
      ))
  )
    throw new Error("INVALID_CLICKHOUSE_URL");
  return {
    url,
    username: process.env.RAG_V2_CLICKHOUSE_USER ?? "default",
    password: process.env.RAG_V2_CLICKHOUSE_PASSWORD ?? "",
    database: process.env.RAG_V2_CLICKHOUSE_DATABASE ?? "rag_v2",
  };
}
export function embeddingProvider(): EmbeddingProvider {
  const dimensions = z.coerce
    .number()
    .int()
    .min(8)
    .max(8192)
    .parse(process.env.RAG_V2_EMBEDDING_DIMENSIONS ?? 1536);
  if (process.env.RAG_V2_EMBEDDING_PROVIDER === "test") {
    if (process.env.RAG_V2_ALLOW_TEST_PROVIDER !== "true")
      throw new Error("TEST_PROVIDER_NOT_ALLOWED");
    return deterministicProvider(dimensions);
  }
  if (
    process.env.RAG_V2_EMBEDDING_PROVIDER &&
    process.env.RAG_V2_EMBEDDING_PROVIDER !== "openai-compatible"
  )
    throw new Error("INVALID_EMBEDDING_PROVIDER");
  const apiKey = process.env.RAG_V2_EMBEDDING_API_KEY;
  if (!apiKey) throw new Error("EMBEDDING_KEY_REQUIRED");
  return openAICompatible({
    endpoint:
      process.env.RAG_V2_EMBEDDING_URL ??
      "https://api.openai.com/v1/embeddings",
    apiKey,
    model: process.env.RAG_V2_EMBEDDING_MODEL ?? "text-embedding-3-small",
    dimensions,
  });
}
