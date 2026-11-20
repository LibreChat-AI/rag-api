import { Hono } from "hono";
import { jwtVerify } from "jose";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ContentfulStatusCode } from "hono/utils/http-status";
import type { Runner } from "./process";
import { Admission } from "./admission";
import { configSchema, type Config } from "./config";
import { ExtractionError, type ErrorCode } from "./contract";
import { runWorker } from "./process";
import { stageUpload } from "./upload";

const status: Record<ErrorCode, ContentfulStatusCode> = {
  EXTRACTION_DISABLED: 404,
  EXTRACTION_AUTH_REQUIRED: 401,
  EXTRACTION_FORBIDDEN: 403,
  UNSUPPORTED_PROFILE: 400,
  UNSUPPORTED_DOCUMENT_TYPE: 415,
  INVALID_MULTIPART: 400,
  PARSER_INPUT_LIMIT: 413,
  PARSER_OUTPUT_LIMIT: 413,
  ZIP_BOMB: 413,
  ARCHIVE_INVALID: 422,
  NO_DOCUMENT_TEXT: 422,
  PARSE_FAILED: 422,
  CONCURRENCY_LIMIT: 429,
  PARSER_CRASH: 503,
  PARSER_UNAVAILABLE: 503,
  PARSER_TIMEOUT: 504,
  REQUEST_CANCELLED: 408,
};

export function createApp(
  input: Partial<Config> = {},
  runner: Runner = runWorker,
): Hono {
  const config = configSchema.parse(input);
  const app = new Hono();
  const admission = new Admission(config.concurrent, config.queued);
  app.onError((error, context) => {
    const code =
      error instanceof ExtractionError ? error.code : "PARSER_UNAVAILABLE";
    return context.json({ detail: { code } }, status[code]);
  });
  app.notFound((context) =>
    context.json({ detail: { code: "EXTRACTION_DISABLED" } }, 404),
  );
  app.get("/health", (context) =>
    context.json({
      status: "UP",
      service: "rag-bun",
      extraction_profiles: config.enabled ? ["document-v1"] : [],
    }),
  );
  if (!config.enabled) return app;
  const key = new TextEncoder().encode(config.secret);
  app.post("/v1/extract", async (context) => {
    const authorization = context.req.header("Authorization");
    if (!authorization?.startsWith("Bearer "))
      throw new ExtractionError("EXTRACTION_AUTH_REQUIRED");
    try {
      const { payload } = await jwtVerify(authorization.slice(7), key, {
        algorithms: ["HS256"],
        issuer: config.issuer,
        audience: config.audience,
        requiredClaims: ["exp", "sub"],
      });
      if (typeof payload.sub !== "string" || !payload.sub)
        throw new ExtractionError("EXTRACTION_AUTH_REQUIRED");
      if (
        !Array.isArray(payload.scopes) ||
        !payload.scopes.includes("rag:documents")
      ) {
        throw new ExtractionError("EXTRACTION_FORBIDDEN");
      }
    } catch (error) {
      if (error instanceof ExtractionError) throw error;
      throw new ExtractionError("EXTRACTION_AUTH_REQUIRED");
    }
    const length = context.req.header("Content-Length");
    if (
      length !== undefined &&
      (!/^\d+$/.test(length) || Number(length) > config.maxBodyBytes)
    ) {
      throw new ExtractionError("PARSER_INPUT_LIMIT");
    }
    const deadline = new AbortController();
    const timer = setTimeout(() => deadline.abort(), config.timeoutMs);
    const signal = AbortSignal.any([context.req.raw.signal, deadline.signal]);
    let release: (() => void) | undefined;
    let directory: string | undefined;
    try {
      release = await admission.acquire(signal);
      signal.throwIfAborted();
      directory = await mkdtemp(
        join(config.tempRoot ?? tmpdir(), "rag-extract-"),
      );
      signal.throwIfAborted();
      const path = join(directory, "input.docx");
      await stageUpload(context.req.raw, path, config, signal);
      const result = await runner(
        {
          path,
          maxOutputBytes: config.maxOutputBytes,
          maxEntryBytes: config.maxEntryBytes,
          maxArchiveBytes: config.maxArchiveBytes,
          maxEntries: config.maxEntries,
        },
        signal,
      );
      signal.throwIfAborted();
      return context.json(result);
    } catch (error) {
      if (deadline.signal.aborted) throw new ExtractionError("PARSER_TIMEOUT");
      if (signal.aborted) throw new ExtractionError("REQUEST_CANCELLED");
      throw error;
    } finally {
      clearTimeout(timer);
      try {
        if (directory) await rm(directory, { recursive: true, force: true });
      } finally {
        release?.();
      }
    }
  });
  return app;
}
