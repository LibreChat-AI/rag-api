import { exportJWK, generateKeyPair, SignJWT } from "jose";
import { mkdir } from "node:fs/promises";

if (process.argv[2] === "prepare") {
  const { publicKey, privateKey } = await generateKeyPair("EdDSA");
  const jwk = {
    ...(await exportJWK(publicKey)),
    kid: "runtime-test",
    alg: "EdDSA",
  };
  const now = Math.floor(Date.now() / 1000);
  const token = await new SignJWT({
    tenant_id: "runtime-test",
    actor_kind: "user",
    grants: [
      {
        namespaceId: "test",
        resourceKind: "document",
        operations: ["read", "write", "delete"],
      },
    ],
  })
    .setProtectedHeader({ kid: "runtime-test", alg: "EdDSA", typ: "JWT" })
    .setSubject("test-user")
    .setIssuer("librechat")
    .setAudience("ragapi")
    .setIssuedAt(now)
    .setNotBefore(now)
    .setExpirationTime(now + 300)
    .setJti(crypto.randomUUID())
    .sign(privateKey);
  await mkdir(".qualification", { recursive: true });
  await Bun.write(".qualification/runtime-token", token);
  // Only public verification material is exported to the service container.
  console.log(JSON.stringify({ keys: [jwk] }));
} else if (process.argv[2] === "check") {
  const base = "http://127.0.0.1:8001";
  const token = await Bun.file(".qualification/runtime-token").text();
  const path = "/v2/namespaces/test/documents/file";
  const headers = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    "Idempotency-Key": "one",
  };
  function check(value: unknown, message: string): asserts value {
    if (!value) throw Error(message);
  }
  const denied = await fetch(base + path);
  check(denied.status === 401, "ANONYMOUS_NOT_DENIED");
  const body = {
    segments: [{ kind: "page", index: 1, text: "runtime alpha original" }],
    original: {
      fileId: "file",
      revision: "original-revision",
      sha256: "a".repeat(64),
      filename: "original.pdf",
      mediaType: "application/pdf",
    },
  };
  const upload = await fetch(base + path, {
    method: "PUT",
    headers,
    body: JSON.stringify(body),
  });
  check(upload.status === 200, "RUNTIME_UPLOAD_FAILED");
  const stored = (await upload.json()) as { document: { generation: string } };
  const search = await fetch(base + "/v2/search", {
    method: "POST",
    headers,
    body: JSON.stringify({
      query: "alpha",
      namespaces: [{ namespaceId: "test" }],
    }),
  });
  const result = (await search.json()) as {
    hits: Array<{
      original: { revision: string };
      actor: string;
      page: number;
    }>;
  };
  check(
    search.ok &&
      result.hits[0]?.original.revision === body.original.revision &&
      result.hits[0].actor === "user:test-user" &&
      result.hits[0].page === 1,
    "RUNTIME_SEARCH_FAILED",
  );
  const context = await fetch(base + path + "/context", { headers });
  check(context.status === 200, "RUNTIME_CONTEXT_FAILED");
  const deleted = await fetch(base + path, {
    method: "DELETE",
    headers: { ...headers, "Idempotency-Key": "delete" },
  });
  check(deleted.status === 204, "RUNTIME_DELETE_FAILED");
  const stale = await fetch(base + path, {
    method: "PUT",
    headers: { ...headers, "Idempotency-Key": "stale" },
    body: JSON.stringify({ ...body, ifMatch: stored.document.generation }),
  });
  check(stale.status === 409, "DELETED_GENERATION_RESURRECTED");
  console.log(
    "PASS: built non-root Bun service, asymmetric auth, real HTTP ingestion/search/context/delete, original revision/provenance, and stale-edit denial.",
  );
} else {
  throw Error("EXPECTED_PREPARE_OR_CHECK");
}
