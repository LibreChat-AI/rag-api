import { z } from "zod";

export const MAX_DOCUMENT_CHUNKS = 10000;

export const id = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/);
export const originalSchema = z
  .object({
    fileId: id,
    revision: id,
    sha256: z.string().regex(/^[a-f0-9]{64}$/),
    filename: z.string().min(1).max(255),
    mediaType: z
      .string()
      .min(1)
      .max(128)
      .regex(/^[\w.+-]+\/[\w.+-]+$/),
  })
  .strict();
export const segmentSchema = z
  .object({
    kind: z.enum(["page", "document"]),
    index: z.number().int().min(1).max(128),
    text: z
      .string()
      .max(1024 * 1024)
      .refine((text) => text.isWellFormed()),
  })
  .strict();
export const ingestSchema = z
  .object({
    title: z.string().max(512).default(""),
    segments: z.array(segmentSchema).min(1).max(128),
    original: originalSchema.nullable().default(null),
    ifMatch: z
      .string()
      .regex(/^[a-f0-9]{32}$/)
      .optional(),
  })
  .strict()
  .superRefine((input, context) => {
    let bytes = 0;
    const first = input.segments[0]!;
    for (const [index, segment] of input.segments.entries()) {
      bytes += Buffer.byteLength(segment.text);
      if (segment.kind !== first.kind || segment.index !== index + 1) {
        context.addIssue({
          code: "custom",
          message: "Segments must be ordered and homogeneous",
        });
      }
    }
    if (first.kind === "document" && input.segments.length !== 1) {
      context.addIssue({
        code: "custom",
        message: "A document has one segment",
      });
    }
    if (
      bytes > 1024 * 1024 ||
      !input.segments.some((segment) => segment.text.trim())
    ) {
      context.addIssue({
        code: "custom",
        message: "Text must be nonempty and at most 1 MiB",
      });
    }
  });
export const searchSchema = z
  .object({
    query: z.string().trim().min(1).max(8192),
    namespaces: z
      .array(
        z
          .object({
            namespaceId: id,
            resourceIds: z.array(id).min(1).max(100).optional(),
          })
          .strict(),
      )
      .min(1)
      .max(8),
    k: z.number().int().min(1).max(100).default(5),
    mode: z.enum(["semantic", "hybrid"]).default("semantic"),
    precision: z.enum(["exact", "quantized"]).default("exact"),
  })
  .strict();
export type IngestInput = z.infer<typeof ingestSchema>;
export type Original = z.infer<typeof originalSchema>;
export type Segment = z.infer<typeof segmentSchema>;
export type SearchInput = z.infer<typeof searchSchema>;
export type Scope = {
  tenantId: string;
  namespaceId: string;
  resourceIds?: readonly string[];
};
export type Document = {
  tenantId: string;
  namespaceId: string;
  fileId: string;
  generation: string;
  version: string;
  state: "ready" | "deleted";
  title: string;
  original: Original | null;
  actor: string;
  sourceClass: "asserted" | "extracted";
  spaceId: string;
  chunkCount: number;
  operationKey: string;
  requestHash: string;
};
export type Chunk = {
  index: number;
  text: string;
  page: number | null;
  segment: number;
  start: number;
  end: number;
  section: string[];
};
export type StoredChunk = Chunk & {
  tenantId: string;
  namespaceId: string;
  fileId: string;
  generation: string;
  spaceId: string;
  actor: string;
  sourceClass: "asserted" | "extracted";
  embedding: readonly number[];
};
export type Hit = Chunk & {
  namespaceId: string;
  fileId: string;
  generation: string;
  distance: number;
  score: number;
  original: Original | null;
  actor: string;
  sourceClass: "asserted" | "extracted";
};
export type Receipt = { requestHash: string; document: Document };
export interface Store {
  get(
    scope: Scope,
    fileId: string,
    signal: AbortSignal,
  ): Promise<Document | null>;
  receipt(
    scope: Scope,
    operationKey: string,
    signal: AbortSignal,
  ): Promise<Receipt | null>;
  insert(
    chunks: readonly StoredChunk[],
    batchId: string,
    signal: AbortSignal,
  ): Promise<void>;
  publish(document: Document, signal: AbortSignal): Promise<void>;
  record(
    scope: Scope,
    operationKey: string,
    receipt: Receipt,
    signal: AbortSignal,
  ): Promise<void>;
  search(
    scopes: readonly Scope[],
    vector: readonly number[],
    spaceId: string,
    input: SearchInput,
    signal: AbortSignal,
  ): Promise<Hit[]>;
  context(
    scope: Scope,
    document: Document,
    signal: AbortSignal,
  ): Promise<Chunk[]>;
  close(): Promise<void>;
}
export class RagError extends Error {
  constructor(
    public readonly code: string,
    public readonly status:
      400 | 401 | 403 | 404 | 409 | 413 | 422 | 429 | 500 | 503 | 504,
  ) {
    super(code);
  }
}

export function narrowScope(scope: Scope, resourceId: string): Scope {
  if (
    scope.resourceIds !== undefined &&
    !scope.resourceIds.includes(resourceId)
  )
    throw new RagError("SCOPE_DENIED", 403);
  return { ...scope, resourceIds: [resourceId] };
}
