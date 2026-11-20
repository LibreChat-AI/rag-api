import { expect, test } from "bun:test";
import { createApp } from "../src/app";
import { auth, MemoryStore, provider, token } from "./helpers";

const path = "/v2/namespaces/library-a/documents/file-a";
const body = {
  segments: [{ kind: "page", index: 1, text: "original alpha document" }],
  original: {
    fileId: "file-a",
    revision: "revision-a",
    sha256: "a".repeat(64),
    filename: "original.pdf",
    mediaType: "application/pdf",
  },
};
async function headers(overrides = {}) {
  return {
    Authorization: `Bearer ${await token(overrides)}`,
    "Content-Type": "application/json",
    "Idempotency-Key": "ingest-a",
  };
}

test("authenticated ingest, search, ordered context, original revision, and deletion", async () => {
  const store = new MemoryStore();
  const app = createApp({ auth, store, provider, writer: true });
  const signed = await headers();
  expect(
    (
      await app.request(path, {
        method: "PUT",
        headers: signed,
        body: JSON.stringify(body),
      })
    ).status,
  ).toBe(200);
  const response = await app.request("/v2/search", {
    method: "POST",
    headers: signed,
    body: JSON.stringify({
      query: "alpha",
      namespaces: [{ namespaceId: "library-a" }],
    }),
  });
  const result = await response.json();
  expect(result.hits[0].original).toEqual(body.original);
  expect(result.hits[0].page).toBe(1);
  expect(result.hits[0].start).toBe(0);
  expect(
    (await app.request(`${path}/context`, { headers: signed })).status,
  ).toBe(200);
  expect(
    (
      await app.request(path, {
        method: "DELETE",
        headers: { ...signed, "Idempotency-Key": "delete-a" },
      })
    ).status,
  ).toBe(204);
  expect((await app.request(path, { headers: signed })).status).toBe(404);
});
test("unauthenticated and unauthorized writes fail before body consumption or I/O", async () => {
  const store = new MemoryStore();
  const app = createApp({ auth, store, provider, writer: true });
  expect(
    (await app.request(path, { method: "PUT", body: "not json" })).status,
  ).toBe(401);
  const signed = await headers({
    grants: [
      {
        namespaceId: "library-a",
        resourceKind: "document",
        operations: ["read"],
        resourceIds: ["file-b"],
      },
    ],
  });
  expect(
    (
      await app.request(path, {
        method: "PUT",
        headers: signed,
        body: "not json",
      })
    ).status,
  ).toBe(403);
  expect(store.calls).toBe(0);
});
test("body fields cannot supply tenant or widen entity scope; API replicas refuse writes", async () => {
  const store = new MemoryStore();
  const signed = await headers();
  const app = createApp({ auth, store, provider, writer: true });
  for (const extra of [{ tenantId: "foreign" }, { entity_id: "victim" }])
    expect(
      (
        await app.request(path, {
          method: "PUT",
          headers: signed,
          body: JSON.stringify({ ...body, ...extra }),
        })
      ).status,
    ).toBe(400);
  expect(
    (
      await createApp({ auth, store, provider }).request(path, {
        method: "PUT",
        headers: signed,
        body: JSON.stringify(body),
      })
    ).status,
  ).toBe(503);
  expect(store.calls).toBe(0);
});
test("search grants restrict files and tenants even when the caller omits file filters", async () => {
  const store = new MemoryStore();
  const app = createApp({ auth, store, provider, writer: true });
  await app.request(path, {
    method: "PUT",
    headers: await headers(),
    body: JSON.stringify(body),
  });
  const signed = await headers({
    grants: [
      {
        namespaceId: "library-a",
        resourceKind: "document",
        operations: ["read"],
        resourceIds: ["file-b"],
      },
    ],
  });
  const request = {
    query: "alpha",
    namespaces: [{ namespaceId: "library-a" }],
  };
  expect(
    (
      await (
        await app.request("/v2/search", {
          method: "POST",
          headers: signed,
          body: JSON.stringify(request),
        })
      ).json()
    ).hits,
  ).toEqual([]);
  const foreign = await headers({ tenant_id: "tenant-b" });
  expect(
    (
      await (
        await app.request("/v2/search", {
          method: "POST",
          headers: foreign,
          body: JSON.stringify(request),
        })
      ).json()
    ).hits,
  ).toEqual([]);
});
test("raw originals use a separate extraction adapter, bind the digest, and publish only validated text", async () => {
  const store = new MemoryStore();
  let calls = 0;
  const app = createApp({
    auth,
    store,
    provider,
    writer: true,
    extractor: {
      async extract() {
        calls++;
        return [{ kind: "page", index: 1, text: "extracted alpha" }];
      },
    },
  });
  const bytes = new TextEncoder().encode("%PDF fixture");
  const digest = new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
  const original = { ...body.original, sha256: digest };
  const signed = {
    ...(await headers()),
    "Content-Type": "application/octet-stream",
    "Content-Length": String(bytes.length),
    "X-Rag-Format": "pdf",
    "X-Rag-Original": JSON.stringify(original),
  };
  expect(
    (
      await app.request(`${path}/content`, {
        method: "PUT",
        headers: signed,
        body: bytes,
      })
    ).status,
  ).toBe(200);
  expect(calls).toBe(1);
  expect(store.chunks[0]!.text).toBe("extracted alpha");
  expect(
    (
      await app.request(`${path}/content`, {
        method: "PUT",
        headers: { ...signed, "X-Rag-Original": JSON.stringify(body.original) },
        body: bytes,
      })
    ).status,
  ).toBe(400);
  expect(calls).toBe(1);
});
