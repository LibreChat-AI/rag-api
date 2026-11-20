import type { Hono } from "hono";
import type { Config } from "./config";
import { createApp } from "./app";
import { fromEnv } from "./config";

export function listen(
  app: Hono,
  config: Config,
  port: number,
  hostname: string,
) {
  return Bun.serve({
    hostname,
    port,
    // Bun includes pending handlers in its idle timer (default: 10 seconds).
    // Its maximum is 255 seconds; config bounds the request deadline to 254.
    idleTimeout: Math.ceil(config.timeoutMs / 1000) + 1,
    maxRequestBodySize: config.maxBodyBytes,
    fetch: app.fetch,
  });
}

if (import.meta.main) {
  try {
    const config = fromEnv(process.env);
    const port = Number(process.env.RAG_PORT ?? 8001);
    if (!Number.isInteger(port) || port < 1 || port > 65535)
      throw new Error("Invalid port");
    const server = listen(
      createApp(config),
      config,
      port,
      process.env.RAG_HOST ?? "0.0.0.0",
    );
    for (const event of ["SIGINT", "SIGTERM"] as const) {
      process.once(event, () => {
        void server.stop(false);
      });
    }
    console.info(`RAG Bun service listening on port ${server.port}`);
  } catch {
    console.error(
      "Unable to start RAG Bun service: check runtime configuration",
    );
    process.exitCode = 1;
  }
}
