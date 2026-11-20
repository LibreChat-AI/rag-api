# Bun/Hono RAG service: first extraction slice

PR #330 starts the **new Bun/Hono service**, under `service/`. It is not an
extension of FastAPI and does not proxy to Python. The existing Python API,
requirements, ingestion, `/text`, collection schema and deployment remain
unchanged. Keep that service available throughout compatibility evaluation.

This first slice implements only `POST /v1/extract` with the `document-v1` DOCX
profile. Retrieval, embeddings, reranking, PDF accounting, OCR and the LibreChat
client migration remain follow-up slices. **Do not point LibreChat's existing
`RAG_API_URL` at this service yet:** it does not implement the legacy endpoints.

## Run independently

Requires Bun 1.4.2. The Node AnyDoc binding is pinned to 0.1.3, matching Marco's
LibreChat PR #14701. The binding runs in a separate, killable Bun process, never
in the serving process. No Node server, Python runtime, database or embedding
provider is required.

```sh
cd service
bun install --frozen-lockfile
bun run typecheck
bun test
bun run start
```

The default port is **8001**, so the existing Python service may stay on 8000.
`GET /health` works without external dependencies. Extraction is disabled by
default, and `/v1/extract` is not registered when off.

Build a separate image from the repository root:

```sh
docker build -f service/Dockerfile -t rag-bun .
```

Opt in at startup with `RAG_EXTRACTION_API_ENABLED=true` and `RAG_JWT_SECRET`
(minimum 32 characters). The secret must be the dedicated RAG signing key, **not
LibreChat's session key**. Each extraction request needs an HS256 JWT with
`sub`, `exp`, issuer `librechat`, audience `rag-api`, and a `scopes` array
containing `rag:documents`. Issuer and audience are configurable. Legacy `{id}`
tokens and inference-only tokens do not authorize this new route. That is a
new-service contract, not a change to the old Python API. Coordinate the
LibreChat token supplier with the strict-auth migration before sending traffic.

## Request and response

Send multipart `profile=document-v1` and one `file`. DOCX MIME is authoritative;
a generic MIME requires a `.docx` filename. No caller-provided path or URL is
read, and no hosted parser, OCR service, or inference provider is called.

Successful response:

```json
{
  "profile": "document-v1",
  "text": "# Extracted Markdown\n",
  "format": "markdown",
  "completeness": "complete",
  "may_omit_content": false,
  "pages_needing_ocr": [],
  "truncated": false,
  "parser": { "name": "anydoc", "version": "0.1.3" }
}
```

An archive with non-thumbnail artwork or embedded objects is marked `partial`
and `may_omit_content=true`. Package relationships and content types identify
artwork even when its filename has no image extension. The root relationship
resolves the main document; a conventional `word/document.xml` path is not
required. This is conservative omission detection, **not a
proof that every source element is inspectable**. Never use partial text or a
preview as complete content-inspection input. No automatic fallback occurs
inside the service, so hard refusals cannot accidentally become paid OCR calls.

`detail.code` classifies failures without source text, tokens, or native error
messages:

| Status | Codes | Meaning |
|---|---|---|
| 400/415 | `INVALID_MULTIPART`, `UNSUPPORTED_PROFILE`, `UNSUPPORTED_DOCUMENT_TYPE` | Malformed or unsupported request |
| 401/403 | `EXTRACTION_AUTH_REQUIRED`, `EXTRACTION_FORBIDDEN` | Missing/invalid service token or missing document scope |
| 404 | `EXTRACTION_DISABLED` | Extraction not registered |
| 413 | `PARSER_INPUT_LIMIT`, `PARSER_OUTPUT_LIMIT`, `ZIP_BOMB` | Hard refusal; do not send those bytes to another parser |
| 422 | `ARCHIVE_INVALID`, `NO_DOCUMENT_TEXT`, `PARSE_FAILED` | Unusable archive or no usable conversion |
| 429 | `CONCURRENCY_LIMIT` | Busy; retry later rather than escalating to OCR |
| 503/504 | `PARSER_UNAVAILABLE`, `PARSER_CRASH`, `PARSER_TIMEOUT` | Infrastructure failure or overall deadline |
| 408 | `REQUEST_CANCELLED` | Cancelled operation; no result retained |

## Limits and lifecycle

Authentication and parser admission happen **before** multipart consumption.
Multipart uploads stream to a private random directory. Both declared length
and actual streamed bytes are bounded; there is no `request.formData()` buffer
or health-check round trip. The Bun listener also enforces the body ceiling.

| Environment variable | Default |
|---|---:|
| `RAG_PORT` / `RAG_HOST` | 8001 / 0.0.0.0 |
| `RAG_EXTRACTION_CONCURRENT` | 2 |
| `RAG_EXTRACTION_QUEUED` | 6 |
| `RAG_EXTRACTION_TIMEOUT_MS` | 30,000 |
| `RAG_EXTRACTION_MAX_FILE_BYTES` | 15 MiB |
| `RAG_EXTRACTION_MAX_BODY_BYTES` | 16 MiB |
| `RAG_EXTRACTION_MAX_OUTPUT_BYTES` | 15 MiB |
| `RAG_EXTRACTION_MAX_ENTRY_BYTES` | 25 MiB |
| `RAG_EXTRACTION_MAX_ARCHIVE_BYTES` | 100 MiB |
| `RAG_EXTRACTION_MAX_ENTRIES` | 4,096 |
| `RAG_EXTRACTION_TEMP_DIR` | operating-system temp directory |
| `RAG_JWT_ISSUER` / `RAG_JWT_AUDIENCE` | librechat / rag-api |

Limits are per service process, not cluster-wide quotas. Queue wait, upload
staging and parsing share one deadline. Cancelled queued work is removed; an
active child is killed and reaped before cleanup and slot reuse. The child
receives no JWT secret or provider credentials, and Bun's automatic dotenv
loading is disabled in it. Both sides cap serialized IPC
output. Actual decompressed entry bytes are checked before native parsing;
metadata-only size claims are not trusted. Graceful shutdown stops accepting
requests and allows active operations to finish within their deadlines.

## Verification and adoption

The Bun corpus uses Marco's structured DOCX at
`fb7bbcd9cf75f4f78ecbd5a8780685c481600be2` and the exact output expected by the
Python prototype. Tests use real native parsing, Hono requests and child
processes, with injected crash/hang/overproduction programs for failure cases.
Typecheck and native tests run in a separate Bun CI job alongside the unchanged
Python jobs.

No traffic cutover or database migration is part of this PR. Next, add a flagged
LibreChat adapter using this result contract, then prove content inspection,
preview/sharing, token scopes, outages and mixed-version behavior end to end.
Keep raw Markdown, rich HTML preview and complete inspection as distinct
contracts. Expand formats or enable reranking only behind their own fidelity
and quality gates. No measured speedup or full service parity is claimed here.

The listener idle timeout is set above the overall extraction deadline so a
parse lasting more than Bun's default ten seconds is not reset prematurely.
`RAG_EXTRACTION_TIMEOUT_MS` may be at most 254,000, leaving one second within
Bun's 255-second listener limit for the result or typed timeout response.
