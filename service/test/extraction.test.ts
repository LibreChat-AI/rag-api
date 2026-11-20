import { afterEach, beforeEach, expect, test } from "bun:test";
import { SignJWT } from "jose";
import { unzipSync, zipSync, strFromU8, strToU8 } from "fflate";
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createApp } from "../src/app";
import { configSchema, fromEnv } from "../src/config";
import { DOCX_TYPE, resultSchema } from "../src/contract";
import { runWorker, type Runner } from "../src/process";

const fixture = await Bun.file(
  new URL("../../tests/fixtures/structured.docx", import.meta.url),
).bytes();
const expectedText =
  "# Quarterly Report\n\nThis document summarizes the results for the period.\n\n## Regional Totals\n\n|  |  |  |\n| --- | --- | --- |\n| Region | Units | Revenue |\n| North | 1200 | 48000 |\n| South | 950 | 38000 |\n| East | 1430 | 57200 |\n\n**Totals are unaudited.**\n";
const secret = "test-rag-key-with-more-than-32-characters";
let tempRoot: string;
beforeEach(async () => {
  tempRoot = await mkdtemp(join(tmpdir(), "rag-bun-test-"));
});
afterEach(async () => {
  await rm(tempRoot, { recursive: true, force: true });
});

async function token(
  options: {
    scopes?: string[];
    key?: string;
    audience?: string;
    expiry?: number;
    subject?: string;
  } = {},
) {
  return new SignJWT({ scopes: options.scopes ?? ["rag:documents"] })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuer("librechat")
    .setAudience(options.audience ?? "rag-api")
    .setSubject(options.subject ?? "owner")
    .setExpirationTime(options.expiry ?? Math.floor(Date.now() / 1000) + 60)
    .sign(new TextEncoder().encode(options.key ?? secret));
}
function app(overrides: Parameters<typeof createApp>[0] = {}, runner?: Runner) {
  return createApp({ enabled: true, secret, tempRoot, ...overrides }, runner);
}
function form(
  bytes: Uint8Array = fixture,
  mime = DOCX_TYPE,
  name = "report.docx",
  profile = "document-v1",
) {
  const body = new FormData();
  body.append("profile", profile);
  body.append("file", new File([new Uint8Array(bytes)], name, { type: mime }));
  return body;
}
async function post(
  application = app(),
  body: BodyInit = form(),
  jwt?: string,
  signal?: AbortSignal,
) {
  return application.request("/v1/extract", {
    method: "POST",
    headers: { Authorization: `Bearer ${jwt ?? (await token())}` },
    body,
    signal,
  });
}
async function code(response: Response, status: number, error: string) {
  expect(response.status).toBe(status);
  expect(await response.json()).toEqual({ detail: { code: error } });
}
async function clean() {
  expect(await readdir(tempRoot)).toEqual([]);
}
function extraEntry(name: string, bytes: Uint8Array) {
  return zipSync({ ...unzipSync(fixture), [name]: bytes });
}

// Same golden output as PR 330's Python prototype and Marco's pinned AnyDoc fixture.
test("real AnyDoc DOCX preserves the exact versioned Markdown contract", async () => {
  const response = await post();
  expect(response.status).toBe(200);
  expect(await response.json()).toEqual({
    profile: "document-v1",
    text: expectedText,
    format: "markdown",
    completeness: "complete",
    may_omit_content: false,
    pages_needing_ocr: [],
    truncated: false,
    parser: { name: "anydoc", version: "0.1.3" },
  });
  await clean();
});
test("renamed DOCX follows MIME while generic DOCX follows filename", async () => {
  for (const [mime, name] of [
    [DOCX_TYPE, "renamed.csv"],
    ["application/octet-stream", "REPORT.DOCX"],
  ] as const) {
    expect((await post(app(), form(fixture, mime, name))).status).toBe(200);
  }
  await clean();
});
test("embedded image or object cannot claim complete inspection", async () => {
  for (const name of ["word/media/scan.png", "word/embeddings/object.bin"]) {
    const response = await post(
      app(),
      form(extraEntry(name, new Uint8Array([1, 2, 3]))),
    );
    expect(response.status).toBe(200);
    const result = resultSchema.parse(await response.json());
    expect(result.completeness).toBe("partial");
    expect(result.may_omit_content).toBe(true);
  }
  await clean();
});
test("image relationships and content types identify non-image filenames", async () => {
  const data = unzipSync(fixture);
  data["word/media/image.bin"] = new Uint8Array([1, 2, 3]);
  data["[Content_Types].xml"] = strToU8(
    strFromU8(data["[Content_Types].xml"]!).replace(
      "</Types>",
      '<Override PartName="/word/media/image.bin" ContentType="image/png"/></Types>',
    ),
  );
  let response = await post(app(), form(zipSync(data)));
  expect(response.status).toBe(200);
  expect(resultSchema.parse(await response.json()).may_omit_content).toBe(true);
  data["[Content_Types].xml"] = unzipSync(fixture)["[Content_Types].xml"]!;
  data["word/_rels/document.xml.rels"] = strToU8(
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="art" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image.bin"/></Relationships>',
  );
  response = await post(app(), form(zipSync(data)));
  expect(response.status).toBe(200);
  expect(resultSchema.parse(await response.json()).completeness).toBe(
    "partial",
  );
  await clean();
});
test("relationship-resolved main documents do not need a word directory", async () => {
  const data = unzipSync(fixture);
  for (const name of Object.keys(data)) {
    if (!name.startsWith("word/")) continue;
    data[name.replace("word/", "content/")] = data[name]!;
    delete data[name];
  }
  for (const name of ["_rels/.rels", "[Content_Types].xml"]) {
    data[name] = strToU8(
      strFromU8(data[name]!).replaceAll("word/", "content/"),
    );
  }
  const response = await post(app(), form(zipSync(data)));
  expect(response.status).toBe(200);
  const result = resultSchema.parse(await response.json());
  expect(result.text).toBe(expectedText);
  await clean();
});
test("external main-part relationships and DTD metadata are hard refusals", async () => {
  const data = unzipSync(fixture);
  data["_rels/.rels"] = strToU8(
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="doc" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="https://example.invalid/main.xml" TargetMode="External"/></Relationships>',
  );
  await code(await post(app(), form(zipSync(data))), 422, "ARCHIVE_INVALID");
  data["_rels/.rels"] = strToU8(
    '<!DOCTYPE Relationships [<!ENTITY x SYSTEM "file:///etc/passwd">]><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">&x;</Relationships>',
  );
  await code(await post(app(), form(zipSync(data))), 422, "ARCHIVE_INVALID");
  await clean();
});
test("cover thumbnail does not cause paid OCR escalation", async () => {
  const response = await post(
    app(),
    form(extraEntry("docProps/thumbnail.png", new Uint8Array([1]))),
  );
  expect(response.status).toBe(200);
  expect((await response.json()).may_omit_content).toBe(false);
  await clean();
});
test("disabled service has no extraction route, even for a malformed body", async () => {
  const response = await post(createApp(), "not multipart");
  await code(response, 404, "EXTRACTION_DISABLED");
  await clean();
  const health = await createApp().request("/health");
  expect(health.status).toBe(200);
  expect((await health.json()).extraction_profiles).toEqual([]);
});
test("enabled startup requires a dedicated service key and finite limits", () => {
  expect(() => createApp({ enabled: true })).toThrow();
  for (const limits of [
    { timeoutMs: Infinity },
    { concurrent: 0 },
    { queued: -1 },
    { maxBodyBytes: 1 },
  ]) {
    expect(() =>
      configSchema.parse({ enabled: true, secret, ...limits }),
    ).toThrow();
  }
  expect(() => fromEnv({ RAG_EXTRACTION_API_ENABLED: "yes" })).toThrow();
  expect(() =>
    fromEnv({ RAG_JWT_SECRET: secret, JWT_SECRET: secret }),
  ).toThrow();
});
test("rejects absent, expired, wrong-audience, and session-key tokens before parsing", async () => {
  const application = app();
  for (const jwt of [
    "",
    await token({ expiry: 1 }),
    await token({ audience: "session" }),
    await token({ key: "different-session-signing-key-32-characters" }),
  ]) {
    await code(
      await post(application, "not multipart", jwt),
      401,
      "EXTRACTION_AUTH_REQUIRED",
    );
  }
  const noHeader = await application.request("/v1/extract", { method: "POST" });
  await code(noHeader, 401, "EXTRACTION_AUTH_REQUIRED");
  await clean();
});
test("inference scopes never grant document extraction", async () => {
  for (const scopes of [[], ["rag:embed"], ["rag:rerank"]]) {
    await code(
      await post(app(), "not multipart", await token({ scopes })),
      403,
      "EXTRACTION_FORBIDDEN",
    );
  }
  await clean();
});
test("legacy id token cannot reach the new service", async () => {
  const legacy = await new SignJWT({ id: "owner" })
    .setProtectedHeader({ alg: "HS256" })
    .sign(new TextEncoder().encode(secret));
  await code(await post(app(), "bad", legacy), 401, "EXTRACTION_AUTH_REQUIRED");
  await clean();
});
test("unsupported type/profile and malformed multipart never reach native parsing", async () => {
  await code(
    await post(app(), form(fixture, "application/pdf")),
    415,
    "UNSUPPORTED_DOCUMENT_TYPE",
  );
  await code(
    await post(app(), form(fixture, "text/markdown", "note.md")),
    415,
    "UNSUPPORTED_DOCUMENT_TYPE",
  );
  await code(
    await post(app(), form(fixture, DOCX_TYPE, "report.docx", "raw-v1")),
    400,
    "UNSUPPORTED_PROFILE",
  );
  await code(await post(app(), "bad"), 400, "INVALID_MULTIPART");
  await clean();
});
test("duplicate fields, extra files and missing profile are rejected", async () => {
  const duplicate = form();
  duplicate.append("profile", "document-v1");
  await code(await post(app(), duplicate), 400, "INVALID_MULTIPART");
  const files = form();
  files.append("file", new File([fixture], "second.docx", { type: DOCX_TYPE }));
  await code(await post(app(), files), 400, "INVALID_MULTIPART");
  const missing = form();
  missing.delete("profile");
  await code(await post(app(), missing), 400, "INVALID_MULTIPART");
  await clean();
});
test("invalid archive errors never contain caller text", async () => {
  await code(
    await post(app(), form(new TextEncoder().encode("private contents"))),
    422,
    "ARCHIVE_INVALID",
  );
  await clean();
});
test("zip bomb, total size and entry counts are hard refusals", async () => {
  const bomb = extraEntry("word/bomb.xml", new Uint8Array(20_000));
  await code(
    await post(app({ maxEntryBytes: 10_000 }), form(bomb)),
    413,
    "ZIP_BOMB",
  );
  await code(await post(app({ maxArchiveBytes: 1000 })), 413, "ZIP_BOMB");
  await code(await post(app({ maxEntries: 2 })), 413, "ZIP_BOMB");
  await clean();
});
test("empty document returns no text instead of a successful extraction", async () => {
  const xml = new TextEncoder().encode(
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p/></w:body></w:document>',
  );
  await code(
    await post(app(), form(extraEntry("word/document.xml", xml))),
    422,
    "NO_DOCUMENT_TEXT",
  );
  await clean();
});
test("file size and serialized native output have independent ceilings", async () => {
  await code(await post(app({ maxFileBytes: 500 })), 413, "PARSER_INPUT_LIMIT");
  await code(
    await post(app({ maxOutputBytes: 100 })),
    413,
    "PARSER_OUTPUT_LIMIT",
  );
  await clean();
});
test("parent caps IPC even if a child ignores its output limit", async () => {
  const runner: Runner = (request, signal) =>
    runWorker(request, signal, [
      process.execPath,
      "-e",
      'process.stdout.write("x".repeat(1000))',
    ]);
  await code(
    await post(app({ maxOutputBytes: 100 }, runner)),
    413,
    "PARSER_OUTPUT_LIMIT",
  );
  await clean();
});
test("native children do not reload secrets from dotenv files", async () => {
  await Bun.write(
    join(tempRoot, ".env"),
    "NATIVE_ENV_CANARY=should-not-reach-parser\n",
  );
  const result = {
    profile: "document-v1",
    text: "isolated",
    format: "markdown",
    completeness: "complete",
    may_omit_content: false,
    pages_needing_ocr: [],
    truncated: false,
    parser: { name: "anydoc", version: "0.1.3" },
  };
  const script = `const result = ${JSON.stringify(result)}; if (process.env.NATIVE_ENV_CANARY) result.text = "leaked"; console.log(JSON.stringify({ok:true,result}));`;
  const response = await runWorker(
    {
      path: "unused",
      maxOutputBytes: 4096,
      maxEntryBytes: 4096,
      maxArchiveBytes: 4096,
      maxEntries: 5,
    },
    new AbortController().signal,
    [process.execPath, "--cwd", tempRoot, "-e", script],
  );
  expect(response.text).toBe("isolated");
});

test("child crash and malformed response are sanitized and permit retry", async () => {
  for (const source of [
    "process.exit(11)",
    'console.log("private native failure")',
    'console.log("null")',
  ]) {
    const runner: Runner = (request, signal) =>
      runWorker(request, signal, [process.execPath, "-e", source]);
    await code(await post(app({}, runner)), 503, "PARSER_CRASH");
    await clean();
  }
  expect((await post()).status).toBe(200);
  await clean();
});

const sleepPath = fileURLToPath(new URL("./sleep.fixture.ts", import.meta.url));
const sleeping: Runner = (request, signal) =>
  runWorker(request, signal, [process.execPath, sleepPath, request.path]);
async function childPid() {
  for (let i = 0; i < 200; i++) {
    for (const dir of await readdir(tempRoot)) {
      try {
        return Number(
          await readFile(join(tempRoot, dir, "input.docx.pid"), "utf8"),
        );
      } catch {
        /* Not started yet. */
      }
    }
    await Bun.sleep(10);
  }
  throw new Error("Child never started");
}
function reaped(pid: number) {
  expect(() => process.kill(pid, 0)).toThrow();
}
test("overall deadline kills and reaps a running native process before cleanup", async () => {
  const pending = post(app({ timeoutMs: 400 }, sleeping));
  const pid = await childPid();
  await code(await pending, 504, "PARSER_TIMEOUT");
  reaped(pid);
  await clean();
});
test("abort kills and reaps the child, cleans temp files and permits the next request", async () => {
  let calls = 0;
  const runner: Runner = (request, signal) =>
    ++calls === 1 ? sleeping(request, signal) : runWorker(request, signal);
  const application = app({}, runner);
  const controller = new AbortController();
  const pending = post(application, form(), await token(), controller.signal);
  const pid = await childPid();
  controller.abort();
  await code(await pending, 408, "REQUEST_CANCELLED");
  reaped(pid);
  await clean();
  expect((await post(application)).status).toBe(200);
  await clean();
});
test("overload refuses before reading or staging another request body", async () => {
  const application = app({ concurrent: 1, queued: 0 }, sleeping);
  const controller = new AbortController();
  const pending = post(application, form(), await token(), controller.signal);
  const pid = await childPid();
  let pulls = 0;
  const body = new ReadableStream(
    {
      pull(control) {
        pulls++;
        control.enqueue(new Uint8Array([1]));
      },
    },
    { highWaterMark: 0 },
  );
  const request = new Request("http://test/v1/extract", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${await token()}`,
      "Content-Type": "multipart/form-data; boundary=test",
    },
    body,
  });
  await code(await application.fetch(request), 429, "CONCURRENCY_LIMIT");
  expect(pulls).toBe(0);
  controller.abort();
  await pending;
  reaped(pid);
  await clean();
});
test("body limit is counted for chunked requests, not trusted Content-Length", async () => {
  const application = app({ maxFileBytes: 100, maxBodyBytes: 200 });
  let pulls = 0;
  let cancelled = false;
  const prefix =
    '--test\r\nContent-Disposition: form-data; name="file"; filename="r.docx"\r\nContent-Type: ' +
    DOCX_TYPE +
    "\r\n\r\n";
  const body = new ReadableStream(
    {
      pull(control) {
        pulls++;
        control.enqueue(
          new TextEncoder().encode(pulls === 1 ? prefix : "x".repeat(256)),
        );
      },
      cancel() {
        cancelled = true;
      },
    },
    { highWaterMark: 0 },
  );
  const request = new Request("http://test/v1/extract", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${await token()}`,
      "Content-Type": "multipart/form-data; boundary=test",
    },
    body,
  });
  await code(await application.fetch(request), 413, "PARSER_INPUT_LIMIT");
  expect(pulls).toBeLessThan(5);
  expect(cancelled).toBe(true);
  await clean();
});
