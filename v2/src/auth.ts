import { importJWK, jwtVerify, decodeProtectedHeader, type JWK } from "jose";
import { z } from "zod";
import { id, RagError, type Scope } from "./contracts";

const grantsSchema = z
  .array(
    z
      .object({
        namespaceId: id,
        resourceKind: z.literal("document"),
        operations: z
          .array(z.enum(["read", "write", "delete"]))
          .min(1)
          .max(3),
        resourceIds: z.array(id).min(1).max(100).optional(),
      })
      .strict(),
  )
  .min(1)
  .max(16);
const claimsSchema = z.object({
  sub: id,
  actor_kind: z.enum(["user", "service"]),
  tenant_id: id,
  jti: z.string().min(1).max(256),
  grants: grantsSchema,
});
export type Principal = z.infer<typeof claimsSchema>;
export interface Authenticator {
  verify(authorization: string | undefined): Promise<Principal>;
}

export async function createAuthenticator(
  jwks: { keys: JWK[] },
  issuer = "librechat",
): Promise<Authenticator> {
  if (!jwks.keys.length || jwks.keys.length > 16)
    throw new Error("INVALID_VERIFICATION_KEYS");
  const keys = new Map<
    string,
    { alg: "EdDSA" | "RS256"; key: Awaited<ReturnType<typeof importJWK>> }
  >();
  for (const jwk of jwks.keys) {
    if (
      !jwk.kid ||
      keys.has(jwk.kid) ||
      jwk.d ||
      jwk.k ||
      (jwk.alg !== "EdDSA" && jwk.alg !== "RS256") ||
      (jwk.alg === "EdDSA" && (jwk.kty !== "OKP" || jwk.crv !== "Ed25519")) ||
      (jwk.alg === "RS256" && jwk.kty !== "RSA")
    )
      throw new Error("INVALID_VERIFICATION_KEYS");
    try {
      keys.set(jwk.kid, {
        alg: jwk.alg === "EdDSA" ? "EdDSA" : "RS256",
        key: await importJWK(jwk, jwk.alg),
      });
    } catch {
      throw new Error("INVALID_VERIFICATION_KEYS");
    }
  }
  return {
    async verify(authorization) {
      if (
        !authorization ||
        authorization.length > 16384 ||
        !/^Bearer [\w.-]+$/.test(authorization)
      )
        throw new RagError("AUTH_INVALID", 401);
      try {
        const token = authorization.slice(7);
        const header = decodeProtectedHeader(token);
        const entry = header.kid ? keys.get(header.kid) : undefined;
        if (!entry || header.alg !== entry.alg || header.typ !== "JWT")
          throw new Error("HEADER");
        const { payload } = await jwtVerify(token, entry.key, {
          issuer,
          audience: "ragapi",
          algorithms: [entry.alg],
          clockTolerance: 5,
          maxTokenAge: 300,
          requiredClaims: [
            "sub",
            "actor_kind",
            "tenant_id",
            "jti",
            "iat",
            "nbf",
            "exp",
            "grants",
          ],
        });
        if (
          typeof payload.iat !== "number" ||
          typeof payload.exp !== "number" ||
          typeof payload.nbf !== "number" ||
          payload.exp <= payload.iat ||
          payload.exp - payload.iat > 300 ||
          payload.nbf > payload.exp
        )
          throw new Error("LIFETIME");
        return claimsSchema.parse(payload);
      } catch {
        throw new RagError("AUTH_INVALID", 401);
      }
    },
  };
}

export function authorize(
  principal: Principal,
  namespaceId: string,
  operation: "read" | "write" | "delete",
  resourceIds?: readonly string[],
): Scope {
  const grants = principal.grants.filter(
    (grant) =>
      grant.namespaceId === namespaceId && grant.operations.includes(operation),
  );
  if (!grants.length) throw new RagError("SCOPE_DENIED", 403);
  const unrestricted = grants.some((grant) => grant.resourceIds === undefined);
  const permitted = new Set(grants.flatMap((grant) => grant.resourceIds ?? []));
  if (resourceIds?.some((file) => !unrestricted && !permitted.has(file)))
    throw new RagError("SCOPE_DENIED", 403);
  return {
    tenantId: principal.tenant_id,
    namespaceId,
    ...(!unrestricted || resourceIds
      ? {
          resourceIds: resourceIds ? [...new Set(resourceIds)] : [...permitted],
        }
      : {}),
  };
}
