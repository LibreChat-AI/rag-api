import { databaseClient } from "./clickhouse";
import { databaseConfig, embeddingProvider } from "./config";
import { migrate } from "./schema";

const config = databaseConfig();
const client = databaseClient(config);
try {
  const provider = embeddingProvider();
  const result = await client.query({
    query:
      "SELECT count() AS supported FROM system.settings WHERE name = 'vector_search_use_quantized_codes'",
    format: "JSONEachRow",
  });
  const rows = await result.json<{ supported: string }>();
  const quantized =
    process.env.RAG_V2_QUANTIZED !== "false" &&
    Number(rows[0]?.supported) > 0 &&
    provider.dimensions % 8 === 0;
  await migrate(client, provider.dimensions, quantized);
  console.log(
    JSON.stringify({
      event: "ragv2.migrated",
      dimensions: provider.dimensions,
      quantized,
    }),
  );
} catch {
  console.error("RAG_V2_MIGRATION_FAILED");
  process.exitCode = 1;
} finally {
  await client.close();
}
