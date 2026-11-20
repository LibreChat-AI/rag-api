import { z } from "zod";
import { ingestSchema, RagError, type Segment } from "./contracts";
import { Gate, readBounded } from "./limits";

export interface Extractor {
  extract(
    bytes: Uint8Array<ArrayBuffer>,
    format: "pdf" | "docx",
    digest: string,
    signal: AbortSignal,
  ): Promise<Segment[]>;
}
const resultSchema = z
  .object({
    version: z.literal(1),
    operation: z.literal("document.extract-text"),
    format: z.enum(["pdf", "docx"]),
    inputSha256: z.string(),
    textBytes: z.number().int().nonnegative(),
    segments: ingestSchema.shape.segments,
  })
  .strict();

export function unixExtractor(socket: string): Extractor {
  if (!socket.startsWith("/") || socket.includes("\0"))
    throw new Error("INVALID_EXTRACTION_SOCKET");
  const gate = new Gate(2);
  return {
    extract: (bytes, format, digest, signal) =>
      gate.run(signal, async () => {
        let response: Response;
        try {
          response = await fetch("http://localhost/v1/extract-text", {
            unix: socket,
            method: "POST",
            redirect: "error",
            signal: AbortSignal.any([signal, AbortSignal.timeout(11000)]),
            headers: {
              "Content-Type": "application/octet-stream",
              "Content-Length": String(bytes.byteLength),
              "X-Extraction-Version": "1",
              "X-Extraction-Format": format,
            },
            body: bytes,
          });
        } catch {
          signal.throwIfAborted();
          throw new RagError("EXTRACTION_UNAVAILABLE", 503);
        }
        const wire = await readBounded(response.body, 3 * 1024 * 1024, signal);
        if (!response.ok) {
          if (response.status === 429 || response.status === 503)
            throw new RagError("EXTRACTION_BUSY", 503);
          if (response.status === 504)
            throw new RagError("EXTRACTION_DEADLINE", 504);
          throw new RagError(
            "EXTRACTION_REJECTED",
            response.status === 413 ? 413 : 422,
          );
        }
        try {
          const result = resultSchema.parse(
            JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(wire)),
          );
          if (
            result.format !== format ||
            result.inputSha256 !== digest ||
            result.textBytes !==
              result.segments.reduce(
                (size, segment) => size + Buffer.byteLength(segment.text),
                0,
              ) ||
            result.segments.some(
              (segment) =>
                segment.kind !== (format === "pdf" ? "page" : "document"),
            )
          )
            throw new Error("INVALID_RESULT");
          return ingestSchema.parse({ segments: result.segments }).segments;
        } catch {
          throw new RagError("INVALID_EXTRACTION_RESULT", 503);
        }
      }),
  };
}
