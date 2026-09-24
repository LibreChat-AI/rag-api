# Opt-in document extraction contract

`POST /v1/extract` implements a **document-v1** profile for DOCX only. It does not
replace `/text`, alter ingestion/chunking, or invoke embeddings, OCR or the vector
store. The route is disabled by default and requires a verified `JWT_SECRET`
token with an `id` claim when enabled. The main RAG application **still initializes
its vector store and embeddings at startup**. A parsing-only deployment remains a
separate migration.

Install the pinned optional engine in a custom image or Python environment,
then opt in explicitly before starting (or restarting) the API. With the flag
off, the route is not registered and does not parse multipart bodies:

```sh
pip install -r requirements.extraction.txt
export RAG_EXTRACTION_API_ENABLED=true
```

Send multipart `file` and `profile=document-v1`. Use DOCX MIME or a `.docx`
filename with a generic MIME. Success includes `text` (Markdown), `format`,
`profile`, `completeness` (`complete`/`partial`), `may_omit_content`,
`pages_needing_ocr` (empty until PDF support), `truncated` (always false), and
`parser: {name, version}`. An embedded image marks the DOCX as *partial*. Never
use partial text as proof of full content inspection. `complete` means the
supported conversion finished without *known* omitted image entries, not that
all information in the source is provably inspectable. No hosted OCR or
second-parser fallback runs inside this endpoint.

Failures use `detail.code`, not the native exception message:

| Status | Codes | Action |
|---|---|---|
| 400/415 | `UNSUPPORTED_PROFILE`, `UNSUPPORTED_DOCUMENT_TYPE` | Select a supported profile/type |
| 401/404 | `EXTRACTION_AUTH_REQUIRED`, `EXTRACTION_DISABLED` | Authenticate/opt in |
| 413 | `PARSER_INPUT_LIMIT`, `PARSER_OUTPUT_LIMIT`, `ZIP_BOMB` | Hard refusal; never send the same bytes to another parser |
| 422 | `ARCHIVE_INVALID`, `NO_DOCUMENT_TEXT`, `PARSE_FAILED` | Unusable archive or empty/unconvertible document |
| 429 | `CONCURRENCY_LIMIT` | Retry later; not a reason to invoke paid OCR |
| 503/504 | `PARSER_UNAVAILABLE`, `PARSER_CRASH`, `PARSER_TIMEOUT` | Retry or fix the service |

The route checks the input limit of 15 MiB while staging the upload;
serialized output is capped at 15 MiB before IPC. Starlette may have already
spooled a multipart upload before the route runs: configure an upstream HTTP
body-size limit as well for internet-facing deployments. The child checks
*actual decompressed* ZIP entry bytes: at most
25 MiB per entry, 100 MiB in total and 4,096 entries. Defaults are two active
parses and six queued per API process. Set `RAG_EXTRACTION_CONCURRENT`,
`RAG_EXTRACTION_QUEUED` and `RAG_EXTRACTION_TIMEOUT_SECONDS` to tune admission
and the overall 30-second default deadline (queue wait, upload staging, parse).
On cancellation/timeout the child is killed and reaped before its temp file is
removed and its slot is reused.

This is the first **service-side** slice. Existing LibreChat local parsing and
RAG `/text` behavior remain in place until cross-service tests establish policy,
authorization, preview, failure and compatibility behavior for each consumer.
The real DOCX test fixture is copied from Marco's LibreChat AnyDoc PR #14701 at
`fb7bbcd9cf75f4f78ecbd5a8780685c481600be2`.
