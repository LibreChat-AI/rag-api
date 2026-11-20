import { z } from "zod";

export const resultSchema = z
  .object({
    profile: z.literal("document-v1"),
    text: z.string().min(1),
    format: z.literal("markdown"),
    completeness: z.enum(["complete", "partial"]),
    may_omit_content: z.boolean(),
    pages_needing_ocr: z.array(z.number().int().positive()),
    truncated: z.literal(false),
    parser: z.object({
      name: z.literal("anydoc"),
      version: z.literal("0.1.3"),
    }),
  })
  .strict()
  .superRefine((result, ctx) => {
    if ((result.completeness === "partial") !== result.may_omit_content) {
      ctx.addIssue({ code: "custom", message: "Inconsistent completeness" });
    }
  });
export type ExtractionResult = z.infer<typeof resultSchema>;

export const errorSchema = z.enum([
  "EXTRACTION_DISABLED",
  "EXTRACTION_AUTH_REQUIRED",
  "EXTRACTION_FORBIDDEN",
  "UNSUPPORTED_PROFILE",
  "UNSUPPORTED_DOCUMENT_TYPE",
  "INVALID_MULTIPART",
  "PARSER_INPUT_LIMIT",
  "PARSER_OUTPUT_LIMIT",
  "ZIP_BOMB",
  "ARCHIVE_INVALID",
  "NO_DOCUMENT_TEXT",
  "PARSE_FAILED",
  "CONCURRENCY_LIMIT",
  "PARSER_CRASH",
  "PARSER_UNAVAILABLE",
  "PARSER_TIMEOUT",
  "REQUEST_CANCELLED",
]);
export type ErrorCode = z.infer<typeof errorSchema>;
export class ExtractionError extends Error {
  constructor(readonly code: ErrorCode) {
    super(code);
  }
}
export const workerResponseSchema = z.discriminatedUnion("ok", [
  z.object({ ok: z.literal(true), result: resultSchema }),
  z.object({ ok: z.literal(false), code: errorSchema }),
]);
export const DOCX_TYPE =
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
