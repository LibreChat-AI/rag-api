import busboy from "busboy";
import { createWriteStream } from "node:fs";
import { basename, extname } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { Config } from "./config";
import { DOCX_TYPE, ExtractionError } from "./contract";

export async function stageUpload(
  request: Request,
  path: string,
  config: Config,
  signal: AbortSignal,
): Promise<void> {
  if (!request.body) throw new ExtractionError("INVALID_MULTIPART");
  let parser: ReturnType<typeof busboy>;
  try {
    parser = busboy({
      headers: { "content-type": request.headers.get("content-type") ?? "" },
      limits: {
        fileSize: config.maxFileBytes,
        files: 1,
        fields: 1,
        parts: 3,
        fieldSize: 64,
      },
    });
  } catch {
    throw new ExtractionError("INVALID_MULTIPART");
  }
  const reader = request.body.getReader();
  const abort = () => {
    void reader.cancel().catch(() => {});
  };
  signal.addEventListener("abort", abort, { once: true });
  let bytes = 0;
  const input = Readable.from(
    (async function* () {
      try {
        while (true) {
          signal.throwIfAborted();
          const part = await reader.read();
          if (part.done) break;
          bytes += part.value.byteLength;
          if (bytes > config.maxBodyBytes)
            throw new ExtractionError("PARSER_INPUT_LIMIT");
          yield part.value;
        }
      } finally {
        // Stop unread HTTP input before releasing the reader's lock.
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
    })(),
  );
  let profile: string | undefined;
  let fileCount = 0;
  let refusal: ExtractionError | undefined;
  const writes: Promise<void>[] = [];
  const fail = (error: ExtractionError) => {
    refusal ??= error;
    parser.destroy(error);
  };
  parser.on("field", (name, value, info) => {
    if (name !== "profile" || info.valueTruncated || profile !== undefined) {
      fail(new ExtractionError("INVALID_MULTIPART"));
    } else profile = value;
  });
  parser.on("file", (name, stream, info) => {
    // Busboy destroys its current file stream when the parser is refused, even
    // before a disk pipeline exists. Refused streams need an error listener too.
    stream.on("error", () => {});
    fileCount++;
    const extension = extname(basename(info.filename)).toLowerCase();
    const generic = [
      "application/octet-stream",
      "binary/octet-stream",
    ].includes(info.mimeType);
    if (name !== "file" || fileCount !== 1) {
      stream.resume();
      fail(new ExtractionError("INVALID_MULTIPART"));
      return;
    }
    if (info.mimeType !== DOCX_TYPE && !(generic && extension === ".docx")) {
      stream.resume();
      fail(new ExtractionError("UNSUPPORTED_DOCUMENT_TYPE"));
      return;
    }
    stream.once("limit", () => fail(new ExtractionError("PARSER_INPUT_LIMIT")));
    const write = pipeline(
      stream,
      createWriteStream(path, { flags: "wx", mode: 0o600 }),
      { signal },
    );
    // Attach rejection handling at creation, not only after the multipart parser
    // finishes, so a storage failure terminates upload and never becomes unhandled.
    writes.push(
      write.catch((error: Error) => {
        parser.destroy(error);
        throw error;
      }),
    );
    void writes.at(-1)?.catch(() => {});
  });
  for (const event of ["filesLimit", "fieldsLimit", "partsLimit"] as const) {
    parser.on(event, () => fail(new ExtractionError("INVALID_MULTIPART")));
  }
  try {
    await pipeline(input, parser, { signal });
    await Promise.all(writes);
    signal.throwIfAborted();
    if (fileCount !== 1 || profile === undefined)
      throw new ExtractionError("INVALID_MULTIPART");
    if (profile !== "document-v1")
      throw new ExtractionError("UNSUPPORTED_PROFILE");
  } catch (error) {
    input.destroy();
    parser.destroy();
    await reader.cancel().catch(() => {});
    await Promise.allSettled(writes);
    if (refusal) throw refusal;
    if (error instanceof ExtractionError) throw error;
    if (signal.aborted) throw new ExtractionError("REQUEST_CANCELLED");
    throw new ExtractionError("INVALID_MULTIPART");
  } finally {
    signal.removeEventListener("abort", abort);
  }
}
