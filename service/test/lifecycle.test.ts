import { expect, test } from "bun:test";
import { SignJWT } from "jose";
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Admission } from "../src/admission";
import { createApp } from "../src/app";
import { configSchema } from "../src/config";
import { listen } from "../src/server";
import { DOCX_TYPE } from "../src/contract";
import { runWorker, type Runner } from "../src/process";

const fixture = await Bun.file(
  new URL("../../tests/fixtures/structured.docx", import.meta.url),
).bytes();
const sleepPath = fileURLToPath(new URL("./sleep.fixture.ts", import.meta.url));
const secret = "lifecycle-test-service-signing-key-32-characters";

async function headers() {
  const token = await new SignJWT({ scopes: ["rag:documents"] })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuer("librechat")
    .setAudience("rag-api")
    .setSubject("owner")
    .setExpirationTime("1m")
    .sign(new TextEncoder().encode(secret));
  return { Authorization: `Bearer ${token}` };
}
function form() {
  const body = new FormData();
  body.append("profile", "document-v1");
  body.append("file", new File([fixture], "report.docx", { type: DOCX_TYPE }));
  return body;
}
async function pid(temp: string) {
  for (let i = 0; i < 200; i++) {
    for (const dir of await readdir(temp)) {
      try {
        return Number(
          await readFile(join(temp, dir, "input.docx.pid"), "utf8"),
        );
      } catch {
        /* Child is not running yet. */
      }
    }
    await Bun.sleep(10);
  }
  throw new Error("Child never started");
}

test("cancelled queued work never consumes a slot and remaining work stays FIFO", async () => {
  const admission = new Admission(1, 2);
  const first = await admission.acquire(new AbortController().signal);
  const cancelled = new AbortController();
  const second = admission.acquire(cancelled.signal);
  let granted = false;
  const third = admission
    .acquire(new AbortController().signal)
    .then((release) => {
      granted = true;
      return release;
    });
  cancelled.abort();
  await expect(second).rejects.toMatchObject({ code: "REQUEST_CANCELLED" });
  expect(granted).toBe(false);
  first();
  const release = await third;
  expect(granted).toBe(true);
  release();
  release();
  const next = await admission.acquire(new AbortController().signal);
  next();
});

test("listener waits beyond ten seconds for the configured extraction deadline", async () => {
  const tempRoot = await mkdtemp(join(tmpdir(), "rag-long-parse-"));
  const config = configSchema.parse({
    enabled: true,
    secret,
    tempRoot,
    timeoutMs: 14_000,
  });
  const runner: Runner = async (request, signal) => {
    await Bun.sleep(11_000);
    return runWorker(request, signal);
  };
  const server = listen(createApp(config, runner), config, 0, "127.0.0.1");
  try {
    const response = await fetch(`http://127.0.0.1:${server.port}/v1/extract`, {
      method: "POST",
      headers: await headers(),
      body: form(),
    });
    expect(response.status).toBe(200);
    expect((await response.json()).text).toContain("Quarterly Report");
  } finally {
    await server.stop(true);
    await rm(tempRoot, { recursive: true, force: true });
  }
}, 20_000);

test("actual HTTP disconnect kills the native child and cleans up before retry", async () => {
  const tempRoot = await mkdtemp(join(tmpdir(), "rag-disconnect-"));
  let calls = 0;
  const runner: Runner = (request, signal) =>
    ++calls === 1
      ? runWorker(request, signal, [process.execPath, sleepPath, request.path])
      : runWorker(request, signal);
  const application = createApp(
    {
      enabled: true,
      secret,
      tempRoot,
      concurrent: 1,
      queued: 0,
      timeoutMs: 3000,
    },
    runner,
  );
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: application.fetch,
  });
  const controller = new AbortController();
  try {
    const pending = fetch(`http://127.0.0.1:${server.port}/v1/extract`, {
      method: "POST",
      headers: await headers(),
      body: form(),
      signal: controller.signal,
    });
    void pending.catch(() => {});
    const child = await pid(tempRoot);
    controller.abort();
    await expect(pending).rejects.toThrow();
    for (let i = 0; i < 200 && (await readdir(tempRoot)).length; i++)
      await Bun.sleep(10);
    expect(await readdir(tempRoot)).toEqual([]);
    expect(() => process.kill(child, 0)).toThrow();
    const retry = await fetch(`http://127.0.0.1:${server.port}/v1/extract`, {
      method: "POST",
      headers: await headers(),
      body: form(),
    });
    expect(retry.status).toBe(200);
    expect((await retry.json()).text).toContain("Quarterly Report");
  } finally {
    controller.abort();
    await server.stop(true);
    await rm(tempRoot, { recursive: true, force: true });
  }
});
