import { expect, test } from "bun:test";
import { authorize, createAuthenticator } from "../src/auth";
import { auth, token } from "./helpers";

test("verifies short-lived asymmetric service identities, not browser/code-api tokens", async () => {
  expect((await auth.verify(`Bearer ${await token()}`)).tenant_id).toBe(
    "tenant-a",
  );
  for (const override of [
    { aud: "codeapi" },
    { iss: "foreign" },
    { tenant_id: undefined },
    { grants: undefined },
    { exp: 1 },
    { exp: Math.floor(Date.now() / 1000) + 301 },
    { nbf: Math.floor(Date.now() / 1000) + 600 },
    { iat: Math.floor(Date.now() / 1000) + 600 },
    { sub: "" },
  ]) {
    await expect(
      auth.verify(`Bearer ${await token(override)}`),
    ).rejects.toMatchObject({ code: "AUTH_INVALID" });
  }
  await expect(auth.verify(undefined)).rejects.toMatchObject({ status: 401 });
  const signed = await token();
  await expect(
    auth.verify(`Bearer ${signed.slice(0, -8)}xxxxxxxx`),
  ).rejects.toMatchObject({ status: 401 });
});
test("scope cannot be widened by file ids, operation, library, or tenant assertions", async () => {
  const principal = await auth.verify(
    `Bearer ${await token({ grants: [{ namespaceId: "library-a", resourceKind: "document", operations: ["read"], resourceIds: ["file-a"] }] })}`,
  );
  expect(authorize(principal, "library-a", "read").resourceIds).toEqual([
    "file-a",
  ]);
  expect(() => authorize(principal, "library-a", "read", ["file-b"])).toThrow(
    "SCOPE_DENIED",
  );
  expect(() => authorize(principal, "library-a", "delete", ["file-a"])).toThrow(
    "SCOPE_DENIED",
  );
  expect(() => authorize(principal, "library-b", "read")).toThrow(
    "SCOPE_DENIED",
  );
});
test("fails closed on missing, private, symmetric, or ambiguous verification keys", async () => {
  for (const keys of [
    [],
    [{ kid: "x", alg: "HS256", kty: "oct", k: "secret" }],
    [{ kid: "x", alg: "EdDSA", kty: "OKP", crv: "Ed25519", d: "private" }],
  ]) {
    await expect(createAuthenticator({ keys })).rejects.toThrow(
      "INVALID_VERIFICATION_KEYS",
    );
  }
});

test("resource kinds and actor kinds cannot be forged or promoted by request fields", async () => {
  await expect(
    auth.verify(`Bearer ${await token({ actor_kind: "admin" })}`),
  ).rejects.toMatchObject({ code: "AUTH_INVALID" });
  await expect(
    auth.verify(
      `Bearer ${await token({ grants: [{ namespaceId: "library-a", resourceKind: "memory", operations: ["read"] }] })}`,
    ),
  ).rejects.toMatchObject({ code: "AUTH_INVALID" });
});
