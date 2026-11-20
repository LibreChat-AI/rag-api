import { z } from "zod";
import { createApp } from "./app";
import { createAuthenticator } from "./auth";
import { ClickHouseStore, databaseClient } from "./clickhouse";
import { databaseConfig, embeddingProvider } from "./config";
import { unixExtractor } from "./extraction";

try {
  const jwks = JSON.parse(process.env.RAG_V2_JWKS_JSON ?? "null");
  if (!jwks || !Array.isArray(jwks.keys))
    throw new Error("AUTH_CONFIGURATION_REQUIRED");
  const auth = await createAuthenticator(
    jwks,
    process.env.RAG_V2_JWT_ISSUER ?? "librechat",
  );
  const provider = embeddingProvider();
  const config = databaseConfig();
  const store = new ClickHouseStore(
    databaseClient(config),
    config.database,
    provider.dimensions,
  );
  await store.rows("SELECT 1", {}, AbortSignal.timeout(5000));
  await store.probe();
  const mode = z
    .enum(["api", "coordinator"])
    .parse(process.env.RAG_V2_MODE ?? "api");
  // Exactly one coordinator per database. API replicas are read-only.
  if (mode === "coordinator" && process.env.RAG_V2_SINGLE_WRITER !== "true")
    throw new Error("SINGLE_WRITER_ACKNOWLEDGMENT_REQUIRED");
  const extractor = process.env.RAG_V2_EXTRACTION_SOCKET
    ? unixExtractor(process.env.RAG_V2_EXTRACTION_SOCKET)
    : undefined;
  const app = createApp({
    auth,
    provider,
    store,
    writer: mode === "coordinator",
    extractor,
  });
  const server = Bun.serve({
    hostname: process.env.RAG_V2_HOST ?? "127.0.0.1",
    port: z.coerce
      .number()
      .int()
      .min(1)
      .max(65535)
      .parse(process.env.RAG_V2_PORT ?? 8001),
    fetch: app.fetch,
    maxRequestBodySize: 10 * 1024 * 1024,
  });
  console.log(
    JSON.stringify({
      event: "ragv2.ready",
      mode,
      port: server.port,
      spaceId: provider.spaceId,
    }),
  );
  let stopping = false;
  const shutdown = async () => {
    if (stopping) return;
    stopping = true;
    const deadline = setTimeout(() => {
      void server.stop(true);
      process.exit(1);
    }, 35000);
    await server.stop(false);
    await store.close();
    clearTimeout(deadline);
  };
  process.once("SIGTERM", () => void shutdown());
  process.once("SIGINT", () => void shutdown());
} catch {
  console.error("RAG_V2_STARTUP_FAILED");
  process.exit(1);
}
