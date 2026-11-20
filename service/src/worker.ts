import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { posix } from "node:path";
import type { Readable } from "node:stream";
import { SaxesParser } from "saxes";
import { open, type Entry } from "yauzl";
import { z } from "zod";
import type { ExtractionResult } from "./contract";
import { ExtractionError } from "./contract";

const requestSchema = z.object({
  path: z.string(),
  maxOutputBytes: z.number().int().positive(),
  maxEntryBytes: z.number().int().positive(),
  maxArchiveBytes: z.number().int().positive(),
  maxEntries: z.number().int().positive(),
});
export type WorkerRequest = z.infer<typeof requestSchema>;
const IMAGE =
  /\.(?:jpe?g|png|gif|tiff?|bmp|webp|jp2|jpx|avif|heic|heif|emf|wmf|svg)$/i;
const PREVIEW = /^(?:docProps|Thumbnails)\//i;

function mainPart(target: string): string {
  // OPC targets are package URIs, never filesystem or network locations.
  const decoded = decodeURIComponent(target);
  if (
    /^[a-z]+:/i.test(decoded) ||
    decoded.includes("\\") ||
    decoded.split("/").includes("..")
  ) {
    throw new ExtractionError("ARCHIVE_INVALID");
  }
  return posix.normalize("/" + decoded).slice(1);
}

function inspectArchive(path: string, limits: WorkerRequest): Promise<boolean> {
  return new Promise((resolve, reject) => {
    open(
      path,
      { lazyEntries: true, validateEntrySizes: true },
      (error, zip) => {
        if (error || !zip) {
          reject(new ExtractionError("ARCHIVE_INVALID"));
          return;
        }
        let settled = false;
        let active: Readable | undefined;
        let total = 0;
        let media = false;
        let main: string | undefined;
        const names = new Set<string>();
        const imageParts = new Set<string>();
        const imageExtensions = new Set<string>();
        const fail = (code: "ARCHIVE_INVALID" | "ZIP_BOMB") => {
          if (settled) return;
          settled = true;
          active?.destroy();
          zip.close();
          reject(new ExtractionError(code));
        };
        zip.on("error", () => fail("ARCHIVE_INVALID"));
        zip.on("end", () => {
          if (settled) return;
          if (!names.has("[Content_Types].xml") || !main || !names.has(main)) {
            fail("ARCHIVE_INVALID");
            return;
          }
          for (const name of names) {
            if (PREVIEW.test(name)) continue;
            if (
              imageParts.has(name) ||
              imageExtensions.has(posix.extname(name).slice(1).toLowerCase())
            )
              media = true;
          }
          settled = true;
          resolve(media);
        });
        if (zip.entryCount > limits.maxEntries) {
          fail("ZIP_BOMB");
          return;
        }
        zip.on("entry", (entry: Entry) => {
          if (settled) return;
          if (names.has(entry.fileName)) {
            fail("ARCHIVE_INVALID");
            return;
          }
          names.add(entry.fileName);
          if (/\/$/.test(entry.fileName)) {
            zip.readEntry();
            return;
          }
          if (
            entry.uncompressedSize > limits.maxEntryBytes ||
            total + entry.uncompressedSize > limits.maxArchiveBytes
          ) {
            fail("ZIP_BOMB");
            return;
          }
          if (
            (IMAGE.test(entry.fileName) && !PREVIEW.test(entry.fileName)) ||
            /\/embeddings\//i.test(entry.fileName)
          )
            media = true;
          const isTypes = entry.fileName === "[Content_Types].xml";
          const isRelationships = entry.fileName.endsWith(".rels");
          let metadata: SaxesParser<{ xmlns: true }> | undefined;
          if (isTypes || isRelationships) {
            metadata = new SaxesParser({ xmlns: true });
            metadata.on("error", () => fail("ARCHIVE_INVALID"));
            // No DTD/entity expansion or external XML resources in the admission guard.
            metadata.on("doctype", () => fail("ARCHIVE_INVALID"));
            metadata.on("opentag", (tag) => {
              const attr = (name: string) => tag.attributes[name]?.value;
              if (
                isTypes &&
                attr("ContentType")?.toLowerCase().startsWith("image/")
              ) {
                const part = attr("PartName");
                const extension = attr("Extension");
                if (part) imageParts.add(mainPart(part));
                if (extension) imageExtensions.add(extension.toLowerCase());
              }
              if (!isRelationships || tag.local !== "Relationship") return;
              const type = attr("Type") ?? "";
              if (/\/(image|oleObject|package)$/.test(type)) media = true;
              if (
                entry.fileName !== "_rels/.rels" ||
                !type.endsWith("/officeDocument")
              )
                return;
              if (
                main ||
                attr("TargetMode") === "External" ||
                !attr("Target")
              ) {
                fail("ARCHIVE_INVALID");
                return;
              }
              main = mainPart(attr("Target")!);
            });
          }
          zip.openReadStream(entry, (streamError, stream) => {
            if (streamError || !stream) {
              fail("ARCHIVE_INVALID");
              return;
            }
            active = stream;
            let bytes = 0;
            let decoder: TextDecoder | undefined;
            stream.on("data", (chunk: Buffer) => {
              if (settled) return;
              bytes += chunk.byteLength;
              total += chunk.byteLength;
              if (
                bytes > limits.maxEntryBytes ||
                total > limits.maxArchiveBytes
              ) {
                fail("ZIP_BOMB");
                return;
              }
              if (!metadata) return;
              try {
                decoder ??= new TextDecoder(
                  chunk[0] === 0xff && chunk[1] === 0xfe
                    ? "utf-16le"
                    : chunk[0] === 0xfe && chunk[1] === 0xff
                      ? "utf-16be"
                      : "utf-8",
                  { fatal: true },
                );
                metadata.write(decoder.decode(chunk, { stream: true }));
              } catch {
                fail("ARCHIVE_INVALID");
              }
            });
            stream.on("error", () => fail("ARCHIVE_INVALID"));
            stream.on("end", () => {
              active = undefined;
              if (settled) return;
              try {
                metadata?.write(decoder?.decode() ?? "").close();
              } catch {
                fail("ARCHIVE_INVALID");
              }
              if (!settled) zip.readEntry();
            });
          });
        });
        zip.readEntry();
      },
    );
  });
}

async function extract(request: WorkerRequest): Promise<ExtractionResult> {
  const media = await inspectArchive(request.path, request);
  // Loading a native binding is itself isolated and follows all refusal guards.
  const require = createRequire(import.meta.url);
  let anydoc: typeof import("@firecrawl/anydoc");
  try {
    anydoc = require("@firecrawl/anydoc");
  } catch {
    throw new ExtractionError("PARSER_UNAVAILABLE");
  }
  const bytes = await readFile(request.path);
  let text: string;
  try {
    text = await anydoc.toMarkdownBytes(
      bytes,
      "docx" as import("@firecrawl/anydoc").Format,
    );
  } catch {
    throw new ExtractionError("PARSE_FAILED");
  }
  if (!text.trim()) throw new ExtractionError("NO_DOCUMENT_TEXT");
  if (Buffer.byteLength(text) > request.maxOutputBytes)
    throw new ExtractionError("PARSER_OUTPUT_LIMIT");
  return {
    profile: "document-v1",
    text,
    format: "markdown",
    completeness: media ? "partial" : "complete",
    may_omit_content: media,
    pages_needing_ocr: [],
    truncated: false,
    parser: { name: "anydoc", version: "0.1.3" },
  };
}

if (import.meta.main) {
  let serialized: string;
  try {
    const request = requestSchema.parse(JSON.parse(await Bun.stdin.text()));
    serialized = JSON.stringify({ ok: true, result: await extract(request) });
    if (Buffer.byteLength(serialized) > request.maxOutputBytes)
      throw new ExtractionError("PARSER_OUTPUT_LIMIT");
  } catch (error) {
    serialized = JSON.stringify({
      ok: false,
      code: error instanceof ExtractionError ? error.code : "PARSE_FAILED",
    });
  }
  await Bun.write(Bun.stdout, serialized);
}
