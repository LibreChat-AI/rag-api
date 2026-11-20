import { generateKeyPair, exportJWK, SignJWT } from "jose";
import { createAuthenticator } from "../src/auth";
import { deterministicProvider } from "../src/embeddings";
import type {
  Chunk,
  Document,
  Hit,
  Receipt,
  Scope,
  SearchInput,
  Store,
  StoredChunk,
} from "../src/contracts";

export const signal = () => AbortSignal.timeout(5000);
export const provider = deterministicProvider(64);
export const keypair = await generateKeyPair("EdDSA");
const publicKey = await exportJWK(keypair.publicKey);
export const auth = await createAuthenticator({
  keys: [{ ...publicKey, kid: "test-key", alg: "EdDSA" }],
});
export async function token(
  overrides: Record<string, unknown> = {},
): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  return new SignJWT({
    tenant_id: "tenant-a",
    grants: [
      {
        namespaceId: "library-a",
        resourceKind: "document",
        operations: ["read", "write", "delete"],
      },
    ],
    iss: "librechat",
    aud: "ragapi",
    sub: "user-a",
    actor_kind: "user",
    iat: now,
    nbf: now,
    exp: now + 60,
    jti: crypto.randomUUID(),
    ...overrides,
  })
    .setProtectedHeader({ alg: "EdDSA", typ: "JWT", kid: "test-key" })
    .sign(keypair.privateKey);
}
export class MemoryStore implements Store {
  readonly documents = new Map<string, Document>();
  readonly chunks: StoredChunk[] = [];
  readonly receipts = new Map<string, Receipt>();
  readonly events: string[] = [];
  failBatch = 0;
  failRecord = false;
  calls = 0;
  private docKey(scope: Scope, fileId: string) {
    return JSON.stringify([scope.tenantId, scope.namespaceId, fileId]);
  }
  async get(scope: Scope, fileId: string): Promise<Document | null> {
    this.calls++;
    return this.documents.get(this.docKey(scope, fileId)) ?? null;
  }
  async receipt(scope: Scope, key: string): Promise<Receipt | null> {
    this.calls++;
    return this.receipts.get(this.docKey(scope, key)) ?? null;
  }
  async insert(chunks: readonly StoredChunk[]): Promise<void> {
    this.calls++;
    this.events.push("insert");
    if (
      this.failBatch &&
      this.events.filter((event) => event === "insert").length ===
        this.failBatch
    )
      throw new Error("insertion failed");
    this.chunks.push(...chunks);
  }
  async publish(document: Document): Promise<void> {
    this.calls++;
    this.events.push("publish");
    this.documents.set(this.docKey(document, document.fileId), document);
  }
  async record(scope: Scope, key: string, receipt: Receipt): Promise<void> {
    this.calls++;
    if (this.failRecord) throw new Error("lost acknowledgment");
    this.receipts.set(this.docKey(scope, key), receipt);
  }
  async context(scope: Scope, document: Document): Promise<Chunk[]> {
    this.calls++;
    return this.chunks.filter(
      (chunk) =>
        chunk.tenantId === scope.tenantId &&
        chunk.namespaceId === scope.namespaceId &&
        chunk.fileId === document.fileId &&
        chunk.generation === document.generation,
    );
  }
  async search(
    scopes: readonly Scope[],
    vector: readonly number[],
    spaceId: string,
    input: SearchInput,
  ): Promise<Hit[]> {
    this.calls++;
    return this.chunks
      .flatMap((chunk) => {
        const scope = scopes.find(
          (scope) =>
            scope.tenantId === chunk.tenantId &&
            scope.namespaceId === chunk.namespaceId &&
            (scope.resourceIds === undefined ||
              scope.resourceIds.includes(chunk.fileId)),
        );
        const doc = scope
          ? this.documents.get(this.docKey(scope, chunk.fileId))
          : null;
        if (
          !doc ||
          doc.state !== "ready" ||
          doc.generation !== chunk.generation ||
          chunk.spaceId !== spaceId
        )
          return [];
        const distance = cosine(chunk.embedding, vector);
        return [
          { ...chunk, distance, score: 1 - distance, original: doc.original },
        ];
      })
      .sort((a, b) => a.distance - b.distance)
      .slice(0, input.k);
  }
  async close(): Promise<void> {}
}
function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! ** 2;
    nb += b[i]! ** 2;
  }
  return 1 - dot / Math.sqrt(na * nb);
}
