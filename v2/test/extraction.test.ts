import { expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "node:http";
import { once } from "node:events";
import { unixExtractor } from "../src/extraction";
import { signal } from "./helpers";

test("Bun extraction client uses exact binary length, validates digest and one-based pages, and rejects mismatches", async () => {
  const dir = await mkdtemp(join(tmpdir(), "ragv2-extract-"));
  const socket = join(dir, "service.sock");
  let wrong = false;
  const server = createServer(async (req, res) => {
    const parts: Buffer[] = [];
    for await (const chunk of req) parts.push(chunk);
    expect(req.headers["content-length"]).toBe("3");
    expect(req.headers["x-extraction-version"]).toBe("1");
    expect(Buffer.concat(parts).toString()).toBe("pdf");
    const result = {
      version: 1,
      operation: "document.extract-text",
      format: "pdf",
      inputSha256: wrong ? "b".repeat(64) : "a".repeat(64),
      textBytes: 5,
      segments: [{ kind: "page", index: 1, text: "alpha" }],
    };
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(result));
  });
  server.listen(socket);
  await once(server, "listening");
  try {
    const extractor = unixExtractor(socket);
    expect(
      (
        await extractor.extract(
          new TextEncoder().encode("pdf"),
          "pdf",
          "a".repeat(64),
          signal(),
        )
      )[0]!.index,
    ).toBe(1);
    wrong = true;
    await expect(
      extractor.extract(
        new TextEncoder().encode("pdf"),
        "pdf",
        "a".repeat(64),
        signal(),
      ),
    ).rejects.toMatchObject({ code: "INVALID_EXTRACTION_RESULT" });
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
    await rm(dir, { recursive: true });
  }
});
