import { Hono } from "hono";
import { z } from "zod";
import { randomUUID, createHash } from "node:crypto";
import { authorize, type Authenticator, type Principal } from "./auth";
import {
  id,
  ingestSchema,
  originalSchema,
  RagError,
  searchSchema,
  type Store,
} from "./contracts";
import { QueryEmbeddings, type EmbeddingProvider } from "./embeddings";
import type { Extractor } from "./extraction";
import { Gate, readBounded } from "./limits";
import { Pipeline } from "./pipeline";
import { queryTag } from "./clickhouse";

type Environment = { Variables: { principal: Principal; signal: AbortSignal } };
export function createApp(options: {
  auth: Authenticator;
  store: Store;
  provider: EmbeddingProvider;
  writer?: boolean;
  extractor?: Extractor;
}) {
  const app = new Hono<Environment>();
  const requests = new Gate(64);
  const searchRequests = new Gate(16);
  const ingestionRequests = new Gate(2);
  const queryEmbeddings = new QueryEmbeddings(options.provider);
  const pipeline = new Pipeline(options.store, options.provider);
  app.get("/health", (context) => context.json({ status: "ok", version: 2 }));
  app.use("/v2/*", async (context, next) => {
    if (context.req.header("X-API-Key") || context.req.header("User-Id"))
      throw new RagError("AMBIGUOUS_AUTH", 400);
    const signal = AbortSignal.any([
      context.req.raw.signal,
      AbortSignal.timeout(30000),
    ]);
    return requests.run(signal, async () => {
      const principal = await options.auth.verify(
        context.req.header("Authorization"),
      );
      context.set("principal", principal);
      context.set("signal", signal);
      const requestId = randomUUID();
      context.header("X-Request-Id", requestId);
      context.header("Cache-Control", "no-store");
      await queryTag.run(
        `${context.req.method} ${context.req.routePath} ${requestId}`,
        next,
      );
    });
  });
  const identity = (namespaceId: string, fileId: string) => {
    id.parse(namespaceId);
    id.parse(fileId);
  };
  const writeKey = (value: string | undefined) =>
    z
      .string()
      .regex(/^[A-Za-z0-9._:-]{1,128}$/)
      .parse(value);
  const checkWriter = () => {
    if (!options.writer) throw new RagError("WRITER_UNAVAILABLE", 503);
  };
  app.put("/v2/namespaces/:namespaceId/documents/:fileId", async (context) => {
    const { namespaceId, fileId } = context.req.param();
    identity(namespaceId, fileId);
    const principal = context.get("principal");
    const scope = authorize(principal, namespaceId, "write", [fileId]);
    const key = writeKey(context.req.header("Idempotency-Key"));
    checkWriter();
    if (
      context.req.header("Content-Type")?.split(";")[0] !== "application/json"
    )
      throw new RagError("INVALID_CONTENT_TYPE", 400);
    return ingestionRequests.run(context.get("signal"), async () => {
      const bytes = await readBounded(
        context.req.raw.body,
        3 * 1024 * 1024,
        context.get("signal"),
      );
      let json: unknown;
      try {
        json = JSON.parse(
          new TextDecoder("utf-8", { fatal: true }).decode(bytes),
        );
      } catch {
        throw new RagError("INVALID_BODY", 400);
      }
      const input = ingestSchema.parse(json);
      if (input.original && input.original.fileId !== fileId)
        throw new RagError("ORIGINAL_MISMATCH", 400);
      const document = await pipeline.ingest(
        scope,
        fileId,
        principal.sub,
        key,
        input,
        context.get("signal"),
        { actorKind: principal.actor_kind, sourceClass: "asserted" },
      );
      return context.json({ document, offsetEncoding: "utf16", pageBase: 1 });
    });
  });
  app.put(
    "/v2/namespaces/:namespaceId/documents/:fileId/content",
    async (context) => {
      const { namespaceId, fileId } = context.req.param();
      identity(namespaceId, fileId);
      const principal = context.get("principal");
      const scope = authorize(principal, namespaceId, "write", [fileId]);
      const key = writeKey(context.req.header("Idempotency-Key"));
      checkWriter();
      if (!options.extractor) throw new RagError("EXTRACTION_UNAVAILABLE", 503);
      if (context.req.header("Content-Type") !== "application/octet-stream")
        throw new RagError("INVALID_CONTENT_TYPE", 400);
      const format = z
        .enum(["pdf", "docx"])
        .parse(context.req.header("X-Rag-Format"));
      const length = z
        .string()
        .regex(/^[1-9]\d{0,7}$/)
        .parse(context.req.header("Content-Length"));
      if (Number(length) > 10 * 1024 * 1024)
        throw new RagError("BODY_LIMIT", 413);
      const originalHeader = context.req.header("X-Rag-Original");
      if (!originalHeader || originalHeader.length > 2048)
        throw new RagError("INVALID_ORIGINAL", 400);
      let original: z.infer<typeof originalSchema>;
      try {
        original = originalSchema.parse(JSON.parse(originalHeader));
      } catch {
        throw new RagError("INVALID_ORIGINAL", 400);
      }
      if (original.fileId !== fileId)
        throw new RagError("ORIGINAL_MISMATCH", 400);
      return ingestionRequests.run(context.get("signal"), async () => {
        const bytes = await readBounded(
          context.req.raw.body,
          10 * 1024 * 1024,
          context.get("signal"),
        );
        const digest = createHash("sha256").update(bytes).digest("hex");
        if (bytes.byteLength !== Number(length) || digest !== original.sha256)
          throw new RagError("ORIGINAL_MISMATCH", 400);
        const segments = await options.extractor!.extract(
          bytes,
          format,
          digest,
          context.get("signal"),
        );
        const input = ingestSchema.parse({ segments, original });
        const document = await pipeline.ingest(
          scope,
          fileId,
          principal.sub,
          key,
          input,
          context.get("signal"),
          { actorKind: principal.actor_kind, sourceClass: "extracted" },
        );
        return context.json({ document, offsetEncoding: "utf16", pageBase: 1 });
      });
    },
  );
  app.post("/v2/search", async (context) =>
    searchRequests.run(context.get("signal"), async () => {
      if (
        context.req.header("Content-Type")?.split(";")[0] !== "application/json"
      )
        throw new RagError("INVALID_CONTENT_TYPE", 400);
      const bytes = await readBounded(
        context.req.raw.body,
        65536,
        context.get("signal"),
      );
      let json: unknown;
      try {
        json = JSON.parse(
          new TextDecoder("utf-8", { fatal: true }).decode(bytes),
        );
      } catch {
        throw new RagError("INVALID_BODY", 400);
      }
      const input = searchSchema.parse(json);
      const scopes = input.namespaces.map((library) =>
        authorize(
          context.get("principal"),
          library.namespaceId,
          "read",
          library.resourceIds,
        ),
      );
      const vector = await queryEmbeddings.get(
        input.query,
        context.get("signal"),
      );
      const hits = await options.store.search(
        scopes,
        vector,
        options.provider.spaceId,
        input,
        context.get("signal"),
      );
      return context.json({
        hits,
        spaceId: options.provider.spaceId,
        offsetEncoding: "utf16",
        pageBase: 1,
      });
    }),
  );
  app.get("/v2/namespaces/:namespaceId/documents/:fileId", async (context) => {
    const { namespaceId, fileId } = context.req.param();
    identity(namespaceId, fileId);
    const scope = authorize(context.get("principal"), namespaceId, "read", [
      fileId,
    ]);
    const document = await options.store.get(
      scope,
      fileId,
      context.get("signal"),
    );
    if (!document || document.state !== "ready")
      throw new RagError("NOT_FOUND", 404);
    return context.json({ document });
  });
  app.get(
    "/v2/namespaces/:namespaceId/documents/:fileId/context",
    async (context) => {
      const { namespaceId, fileId } = context.req.param();
      identity(namespaceId, fileId);
      const scope = authorize(context.get("principal"), namespaceId, "read", [
        fileId,
      ]);
      const document = await options.store.get(
        scope,
        fileId,
        context.get("signal"),
      );
      if (!document || document.state !== "ready")
        throw new RagError("NOT_FOUND", 404);
      const chunks = await options.store.context(
        scope,
        document,
        context.get("signal"),
      );
      return context.json({
        document,
        chunks,
        offsetEncoding: "utf16",
        pageBase: 1,
      });
    },
  );
  app.delete(
    "/v2/namespaces/:namespaceId/documents/:fileId",
    async (context) => {
      const { namespaceId, fileId } = context.req.param();
      identity(namespaceId, fileId);
      const principal = context.get("principal");
      const scope = authorize(principal, namespaceId, "delete", [fileId]);
      const key = writeKey(context.req.header("Idempotency-Key"));
      checkWriter();
      await pipeline.delete(
        scope,
        fileId,
        principal.sub,
        key,
        context.get("signal"),
        principal.actor_kind,
      );
      return context.body(null, 204);
    },
  );
  app.onError((error, context) => {
    if (error instanceof RagError) {
      if (error.status === 503) context.header("Retry-After", "1");
      return context.json({ error: { code: error.code } }, error.status);
    }
    if (error instanceof z.ZodError)
      return context.json({ error: { code: "INVALID_REQUEST" } }, 400);
    if (context.get("signal")?.aborted)
      return context.json({ error: { code: "DEADLINE" } }, 504);
    console.error(
      JSON.stringify({
        event: "ragv2.request_failed",
        requestId: context.res.headers.get("X-Request-Id"),
      }),
    );
    return context.json({ error: { code: "BACKEND_UNAVAILABLE" } }, 503);
  });
  return app;
}
