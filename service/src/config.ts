import { z } from "zod";

const positive = z.number().int().positive();
export const configSchema = z
  .object({
    enabled: z.boolean().default(false),
    secret: z.string().min(32).optional(),
    issuer: z.string().min(1).default("librechat"),
    audience: z.string().min(1).default("rag-api"),
    concurrent: positive.default(2),
    queued: z.number().int().nonnegative().default(6),
    timeoutMs: positive.max(254_000).default(30_000),
    maxFileBytes: positive.default(15 * 1024 * 1024),
    maxBodyBytes: positive.default(16 * 1024 * 1024),
    maxOutputBytes: positive.default(15 * 1024 * 1024),
    maxEntryBytes: positive.default(25 * 1024 * 1024),
    maxArchiveBytes: positive.default(100 * 1024 * 1024),
    maxEntries: positive.default(4096),
    tempRoot: z.string().min(1).optional(),
  })
  .superRefine((config, ctx) => {
    if (config.enabled && !config.secret) {
      ctx.addIssue({
        code: "custom",
        message: "Enabled extraction requires RAG_JWT_SECRET (32+ characters)",
      });
    }
    if (config.maxBodyBytes <= config.maxFileBytes) {
      ctx.addIssue({
        code: "custom",
        message:
          "Body ceiling must exceed the file ceiling for multipart framing",
      });
    }
  });
export type Config = z.infer<typeof configSchema>;

export function fromEnv(env: NodeJS.ProcessEnv): Config {
  if (env.RAG_JWT_SECRET && env.RAG_JWT_SECRET === env.JWT_SECRET) {
    throw new Error("RAG_JWT_SECRET must differ from JWT_SECRET");
  }
  const number = (name: string) =>
    env[name] === undefined ? undefined : Number(env[name]);
  const enabled = env.RAG_EXTRACTION_API_ENABLED;
  if (
    enabled !== undefined &&
    !["true", "false", "1", "0"].includes(enabled.toLowerCase())
  ) {
    throw new Error("RAG_EXTRACTION_API_ENABLED must be true, false, 1 or 0");
  }
  return configSchema.parse({
    enabled: enabled === "1" || enabled?.toLowerCase() === "true",
    secret: env.RAG_JWT_SECRET,
    issuer: env.RAG_JWT_ISSUER,
    audience: env.RAG_JWT_AUDIENCE,
    concurrent: number("RAG_EXTRACTION_CONCURRENT"),
    queued: number("RAG_EXTRACTION_QUEUED"),
    timeoutMs: number("RAG_EXTRACTION_TIMEOUT_MS"),
    maxFileBytes: number("RAG_EXTRACTION_MAX_FILE_BYTES"),
    maxBodyBytes: number("RAG_EXTRACTION_MAX_BODY_BYTES"),
    maxOutputBytes: number("RAG_EXTRACTION_MAX_OUTPUT_BYTES"),
    maxEntryBytes: number("RAG_EXTRACTION_MAX_ENTRY_BYTES"),
    maxArchiveBytes: number("RAG_EXTRACTION_MAX_ARCHIVE_BYTES"),
    maxEntries: number("RAG_EXTRACTION_MAX_ENTRIES"),
    tempRoot: env.RAG_EXTRACTION_TEMP_DIR,
  });
}
