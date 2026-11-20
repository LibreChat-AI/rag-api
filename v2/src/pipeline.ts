import { createHash, randomBytes } from "node:crypto";
import { chunks, documentEmbeddingInput } from "./chunking";
import {
  RagError,
  narrowScope,
  type Document,
  type IngestInput,
  type Scope,
  type Store,
  type StoredChunk,
} from "./contracts";
import { DocumentLocks, Gate } from "./limits";
import { validateVectors, type EmbeddingProvider } from "./embeddings";

export class Pipeline {
  private readonly locks = new DocumentLocks();
  private readonly admission = new Gate(2);
  private lastVersion = 0n;
  constructor(
    private readonly store: Store,
    private readonly provider: EmbeddingProvider,
  ) {}
  private version(floor = "0"): string {
    if (BigInt(floor) > this.lastVersion) this.lastVersion = BigInt(floor);
    const now = BigInt(Date.now()) * 1000000n;
    this.lastVersion = now > this.lastVersion ? now : this.lastVersion + 1n;
    return this.lastVersion.toString();
  }
  private operation(
    scope: Scope,
    fileId: string,
    subject: string,
    key: string,
    operation: string,
  ): string {
    return createHash("sha256")
      .update(
        JSON.stringify([
          scope.tenantId,
          scope.namespaceId,
          fileId,
          subject,
          key,
          operation,
        ]),
      )
      .digest("hex");
  }
  private async replay(
    scope: Scope,
    fileId: string,
    operationKey: string,
    requestHash: string,
    signal: AbortSignal,
  ): Promise<{ replay: Document | null; current: Document | null }> {
    const receipt = await this.store.receipt(scope, operationKey, signal);
    if (receipt) {
      if (receipt.requestHash !== requestHash)
        throw new RagError("IDEMPOTENCY_CONFLICT", 409);
      return { replay: receipt.document, current: null };
    }
    // Recover a lost acknowledgment between publication and the receipt insert.
    const current = await this.store.get(scope, fileId, signal);
    if (current?.operationKey === operationKey) {
      if (current.requestHash !== requestHash)
        throw new RagError("IDEMPOTENCY_CONFLICT", 409);
      await this.store.record(
        scope,
        operationKey,
        { requestHash, document: current },
        signal,
      );
      return { replay: current, current };
    }
    if (current) {
      // A later revision must not overwrite the only receipt of a published write.
      await this.store.record(
        scope,
        current.operationKey,
        { requestHash: current.requestHash, document: current },
        signal,
      );
    }
    return { replay: null, current };
  }
  async ingest(
    scope: Scope,
    fileId: string,
    subject: string,
    key: string,
    input: IngestInput,
    signal: AbortSignal,
    provenance: {
      actorKind: "user" | "service";
      sourceClass: "asserted" | "extracted";
    } = { actorKind: "user", sourceClass: "asserted" },
  ): Promise<Document> {
    scope = narrowScope(scope, fileId);
    return this.admission.run(signal, () =>
      this.locks.run(
        JSON.stringify([scope.tenantId, scope.namespaceId, fileId]),
        async () => {
          const operationKey = this.operation(
            scope,
            fileId,
            `${provenance.actorKind}:${subject}`,
            key,
            "put",
          );
          const requestHash = createHash("sha256")
            .update(
              JSON.stringify([
                input,
                provenance.sourceClass,
                this.provider.spaceId,
                "chunk-1500-overlap150-budgeted-prefix-v2",
              ]),
            )
            .digest("hex");
          const { replay, current } = await this.replay(
            scope,
            fileId,
            operationKey,
            requestHash,
            signal,
          );
          if (replay) return replay;
          if (
            input.ifMatch &&
            (current?.state !== "ready" || current.generation !== input.ifMatch)
          )
            throw new RagError("PRECONDITION_FAILED", 409);
          const generation = randomBytes(16).toString("hex");
          let chunkCount = 0;
          let batchIndex = 0;
          let batch: ReturnType<typeof chunks> extends Generator<infer T>
            ? T[]
            : never = [];
          let bytes = 0;
          let pendingInsert = Promise.resolve();
          let insertionError: unknown;
          const flush = async () => {
            if (!batch.length) return;
            signal.throwIfAborted();
            if (insertionError) throw insertionError;
            const embeddings = await this.provider.embedDocuments(
              batch.map((chunk) =>
                documentEmbeddingInput(
                  chunk,
                  input.title,
                  this.provider.maxInputBytes,
                ),
              ),
              signal,
            );
            validateVectors(embeddings, batch.length, this.provider.dimensions);
            const rows: StoredChunk[] = batch.map((chunk, index) => ({
              ...chunk,
              tenantId: scope.tenantId,
              namespaceId: scope.namespaceId,
              fileId,
              generation,
              spaceId: this.provider.spaceId,
              actor: `${provenance.actorKind}:${subject}`,
              sourceClass: provenance.sourceClass,
              embedding: embeddings[index]!,
            }));
            await pendingInsert;
            if (insertionError) throw insertionError;
            pendingInsert = this.store
              .insert(rows, `${generation}:${batchIndex++}`, signal)
              .catch((error) => {
                insertionError = error;
              });
            chunkCount += batch.length;
            batch = [];
            bytes = 0;
          };
          try {
            for (const chunk of chunks(input.segments)) {
              const rowBytes =
                Buffer.byteLength(chunk.text) +
                this.provider.dimensions * 24 +
                2048;
              if (
                batch.length &&
                (batch.length >= 32 || bytes + rowBytes > 1024 * 1024)
              )
                await flush();
              batch.push(chunk);
              bytes += rowBytes;
            }
            await flush();
          } finally {
            await pendingInsert;
          }
          if (insertionError) throw insertionError;
          if (!chunkCount) throw new RagError("EMPTY_DOCUMENT", 422);
          signal.throwIfAborted();
          const document: Document = {
            tenantId: scope.tenantId,
            namespaceId: scope.namespaceId,
            fileId,
            generation,
            version: this.version(current?.version),
            state: "ready",
            title: input.title,
            original: input.original,
            actor: `${provenance.actorKind}:${subject}`,
            sourceClass: provenance.sourceClass,
            spaceId: this.provider.spaceId,
            chunkCount,
            operationKey,
            requestHash,
          };
          await this.store.publish(document, signal);
          await this.store.record(
            scope,
            operationKey,
            { requestHash, document },
            signal,
          );
          return document;
        },
      ),
    );
  }
  async delete(
    scope: Scope,
    fileId: string,
    subject: string,
    key: string,
    signal: AbortSignal,
    actorKind: "user" | "service" = "user",
  ): Promise<Document> {
    scope = narrowScope(scope, fileId);
    return this.admission.run(signal, () =>
      this.locks.run(
        JSON.stringify([scope.tenantId, scope.namespaceId, fileId]),
        async () => {
          const operationKey = this.operation(
            scope,
            fileId,
            `${actorKind}:${subject}`,
            key,
            "delete",
          );
          const requestHash = createHash("sha256")
            .update("delete-v1")
            .digest("hex");
          const { replay, current } = await this.replay(
            scope,
            fileId,
            operationKey,
            requestHash,
            signal,
          );
          if (replay) return replay;
          if (!current) throw new RagError("NOT_FOUND", 404);
          const document: Document = {
            ...current,
            state: "deleted",
            actor: `${actorKind}:${subject}`,
            version: this.version(current?.version),
            operationKey,
            requestHash,
          };
          await this.store.publish(document, signal);
          await this.store.record(
            scope,
            operationKey,
            { requestHash, document },
            signal,
          );
          return document;
        },
      ),
    );
  }
}
