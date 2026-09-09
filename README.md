# ID-based RAG FastAPI

## Overview
This project integrates Langchain with FastAPI in an Asynchronous, Scalable manner, providing a framework for document indexing and retrieval, using PostgreSQL/pgvector.

Files are organized into embeddings by `file_id`. The primary use case is for integration with [LibreChat](https://librechat.ai), but this simple API can be used for any ID-based use case.

The main reason to use the ID approach is to work with embeddings on a file-level. This makes for targeted queries when combined with file metadata stored in a database, such as is done by LibreChat.

The API will evolve over time to employ different querying/re-ranking methods, embedding models, and vector stores.

## Features
- **Document Management**: Methods for adding, retrieving, and deleting documents.
- **Vector Store**: Utilizes Langchain's vector store for efficient document retrieval.
- **Asynchronous Support**: Offers async operations for enhanced performance.

## Retrieval scope

Chunks are owned. Every route that reads or removes stored content resolves the
caller's owner set from the verified token and puts it into the store query
*before* ranking, so a chunk outside that set is never read into the process.
The owner set is built in one place — `app/scope.py` — rather than re-derived per
route.

Before this release these routes addressed the store by caller-supplied
`file_id` alone, or authorized a whole result set from the first hit returned:

- `GET /ids` listed every file id in the deployment.
- `POST /query_multiple` performed no authorization at all, so pairing it with
  `GET /ids` disclosed the content of every file to any authenticated caller.
- `POST /query` authorized the whole result set from `documents[0]`, so any hit
  behind the first was never checked. A `file_id` is chosen by whoever uploads,
  so an attacker's own row ranking first authorized the rows behind it.
- `GET /documents`, `GET /documents/{id}/context` and `DELETE /documents` read or
  deleted the chunks of any file id the caller could name.
- A chunk with no recorded `user_id` read as "belongs to everyone".
- On the synchronous store path, a failed ingestion rolled back by `file_id`
  alone, so an upload under someone else's file id destroyed their chunks. The
  async pgvector pipeline already scopes its rollback to the ingestion attempt.

**What changes for callers.** A caller reads and deletes only what it owns. A
file id outside the caller's scope answers "not found" rather than "found but
refused", so none of these routes is an existence oracle. Chunks with no
`user_id` are owned by nobody and are no longer readable — if a deployment holds
such rows and still needs them, stamp an owner on them before upgrading:

```sql
UPDATE langchain_pg_embedding
SET cmetadata = jsonb_set(cmetadata, '{user_id}', '"<owner>"')
WHERE cmetadata->>'user_id' IS NULL;
```

**If this deployment ever ran without `JWT_SECRET`, check for `public` too.** With
no signing key configured there is no caller identity to record, so every chunk
written in that period is owned by the literal string `public`. Once a signing
key is set, callers arrive with their own ids and none of them owns `public`, so
that content stops being readable. Routes other than `/query` returned it to
everybody before this release, which is exactly the hole being closed — but if
the content is still wanted, give it a real owner first:

```sql
-- inspect before rewriting: this is content nobody was ever identified as owning
SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'user_id' = 'public';
```

Deployments that never set `JWT_SECRET` are unaffected: with no key configured
the read scope is `public` as well, so what was written is what is read.

`atlas-mongo` deployments must add `user_id` to the vector search index first;
see [Use Atlas MongoDB as Vector Database](#use-atlas-mongodb-as-vector-database).

**Deleting entity-owned files requires `entity_id`.** Chunks embedded under an
`entity_id` — an agent knowledge base, for instance — are owned by that entity
rather than by the uploading user, so `DELETE /documents` needs the same
`entity_id` that the upload used, as a query parameter alongside the JSON body of
file ids. A delete that omits it resolves to the caller's own scope, matches
nothing, and answers 404 with the chunks left in place. Because a 404 is
indistinguishable from "already deleted", a caller that treats it as success will
orphan those chunks silently.

**Upgrade the client first.** Deploy order matters, in one direction only:

- A client that sends `entity_id` against an *older* build is inert — the
  parameter is simply undeclared there, so the request behaves exactly as before.
- An *older* client against this build orphans every agent knowledge-base file it
  tries to delete.

So upgrade the client first, or both together — never this service first.
LibreChat carries the matching change: it records the owner each embed was made
under and sends it on delete, with `npm run migrate:embed-owners` to backfill
files embedded before that.

**`entity_id` is unchanged and still caller-asserted.** Agent knowledge bases are
owned by an agent id rather than a user id, so a caller reading one names it via
`entity_id`. That id now *widens* the owner set rather than replacing the
caller's identity — the caller's own scope always remains — but nothing in a
token minted today proves the caller may act for the entity it names. A caller
that knows another owner's id can still name it — on read, to reach that owner's
chunks, and on the ingestion routes, where `entity_id` is what gets stamped as the
owner, to write into that owner's namespace. Deployments exposing this API to
untrusted callers must continue to authorize entity access upstream. Closing this
requires the token to carry the entity authorization, which is a coordinated
change with the callers that mint those tokens and is tracked separately from
this release.

## Setup

### Getting Started

- **Configure `.env` file based on [section below](#environment-variables)**
- **Setup pgvector database:**
  - Run an existing PSQL/PGVector setup, or,
  - Docker: `docker compose up` (also starts RAG API)
    - or, use docker just for DB: `docker compose -f ./db-compose.yaml up`
- **Run API**:
  - Docker: `docker compose up` (also starts PSQL/pgvector)
    - or, use docker just for RAG API: `docker compose -f ./api-compose.yaml up`
  - Local:
    - Make sure to setup `DB_HOST` to the correct database hostname
    - Run the following commands (preferably in a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/))
```bash
pip install -r requirements.txt
uvicorn main:app
```

### Clean Install (Local Development)

To do a clean reinstall of all dependencies (e.g., after updating `requirements.txt`):

```bash
# Remove existing virtual environment and recreate it
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For the lite version (without sentence_transformers/huggingface):

```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.lite.txt
```

For Docker, rebuild without cache:

```bash
docker compose build --no-cache
```

### Environment Variables

The following environment variables are required to run the application:

- `RAG_OPENAI_API_KEY`: The API key for OpenAI API Embeddings (if using default settings).
    - Note: `OPENAI_API_KEY` will work but `RAG_OPENAI_API_KEY` will override it in order to not conflict with LibreChat setting.
- `RAG_OPENAI_BASEURL`: (Optional) The base URL for your OpenAI API Embeddings
- `RAG_OPENAI_PROXY`: (Optional) Proxy for OpenAI API Embeddings
    - Note: When using with LibreChat, you can also set `HTTP_PROXY` and `HTTPS_PROXY` environment variables in the `docker-compose.override.yml` file (see [Proxy Configuration](#proxy-configuration) section below)
- `VECTOR_DB_TYPE`: (Optional) select vector database type, default to `pgvector`.
- `POSTGRES_USE_UNIX_SOCKET`: (Optional) Set to "True" when connecting to the PostgreSQL database server with Unix Socket.
- `POSTGRES_DB`: (Optional) The name of the PostgreSQL database, used when `VECTOR_DB_TYPE=pgvector`.
- `POSTGRES_USER`: (Optional) The username for connecting to the PostgreSQL database.
- `POSTGRES_PASSWORD`: (Optional) The password for connecting to the PostgreSQL database.
- `DB_HOST`: (Optional) The hostname or IP address of the PostgreSQL database server.
- `DB_PORT`: (Optional) The port number of the PostgreSQL database server.
- `PGVECTOR_CREATE_EXTENSION`: (Optional) Set to "False" to skip the `CREATE EXTENSION IF NOT EXISTS vector` call on startup. Default is "True". Use this when the `vector` extension is already installed on a managed Postgres (e.g. RDS, Azure Database for PostgreSQL) and the application user is not a superuser.
- `PG_POOL_PRE_PING`: (Optional) Set to "False" to disable SQLAlchemy's pre-ping check. Default is "True". When enabled, the connection pool issues a lightweight `SELECT 1` before handing out a pooled connection, so stale connections dropped by a remote server or middlebox idle timeout are transparently replaced instead of surfacing as query errors. Recommended for any deployment that connects to a remote PostgreSQL instance (managed Postgres, connections that traverse a load balancer, etc.).
- `PG_POOL_RECYCLE`: (Optional) Maximum age in seconds of a pooled connection before it is recycled. Default is "-1" (disabled). Set to a positive value when the server enforces a hard idle or max-lifetime limit (e.g. "1800" for a 30-minute cap).
- `POSTGRES_SCHEMA`: (Optional) Prepend this schema to the Postgres `search_path` so langchain's pgvector tables live in (and are read from) it. Unset by default (uses the user's default schema, typically `public`). Useful when sharing a database with other services — create the schema out-of-band first (`CREATE SCHEMA IF NOT EXISTS <name>; GRANT USAGE, CREATE ON SCHEMA <name> TO <app_user>;`); the RAG API will not create it for you and fails fast at startup if the schema is missing. `public` is always appended to the resulting search path so the `vector` data type stays resolvable when the extension was installed there (the common case). Multiple schemas may be supplied as a comma-separated list (e.g. `myapp,extensions`) when the `vector` extension lives in a non-`public` schema.
- `PGVECTOR_CREATE_LEGACY_INDEXES`: (Optional) Set to "True" to create the legacy `custom_id` and `cmetadata->>'file_id'` indexes on startup. Default is "False".
- `PGVECTOR_MIGRATE_CMETADATA_JSONB`: (Optional) Set to "True" to migrate `langchain_pg_embedding.cmetadata` from JSON to JSONB on startup. Default is "False".
- `PGVECTOR_CREATE_CMETADATA_GIN_INDEX`: (Optional) Set to "True" to create the `cmetadata` JSONB GIN index on startup. Default is "False". The index is created only when `cmetadata` is already JSONB; for a legacy JSON column, also enable `PGVECTOR_MIGRATE_CMETADATA_JSONB` or the index step is skipped.
- `RAG_HOST`: (Optional) The hostname or IP address where the API server will run. Defaults to "0.0.0.0"
- `RAG_PORT`: (Optional) The port number where the API server will run. Defaults to port 8000.
- `JWT_SECRET`: (Optional) The secret key used for verifying JWT tokens for requests.
  - The secret is only used for verification. This basic approach assumes a signed JWT from elsewhere.
  - Omit to run API without requiring authentication

- `COLLECTION_NAME`: (Optional) The name of the collection in the vector store. Default value is "testcollection".
- `CHUNK_SIZE`: (Optional) The size of the chunks for text processing. Default value is "1500".
- `CHUNK_OVERLAP`: (Optional) The overlap between chunks during text processing. Default value is "100".
- `EMBEDDING_BATCH_SIZE`: (Optional) Number of document chunks to process per batch. Defaults to `500`; set to `0` to disable batching. Recommended value is `750` for `text-embedding-3-small`.
- `EMBEDDING_MAX_QUEUE_SIZE`: (Optional) Maximum number of batches to buffer in memory during async processing. Default value is "3".
- `PARALLEL_EXECUTION`: (Optional) Maximum number of async embedding/database insertion consumers to run per file when batching is enabled. Default value is "2".
- `RAG_INGESTION_WINDOW_SIZE`: (Optional) Maximum prepared chunks retained between parsing and insertion. Defaults to `100` and remains independent from the provider batch size.
- `RAG_INGESTION_CONCURRENCY`: (Optional) Maximum complete file ingestions processed concurrently per API process. Defaults to `2`; additional uploads wait without starting parser or embedding work.
- `RAG_THREAD_POOL_SIZE`: (Optional) Maximum worker threads shared by parsing and vector-store operations. Defaults to the detected CPU count, capped at `8`.
- `RAG_DISTANCE_THRESHOLD`: (Optional, `VECTOR_DB_TYPE=pgvector` only) Drop results whose vector distance is greater than this value, after the top-`k` search. Unset by default (no filtering). Lower distance = more similar, so e.g. `0.5` keeps only hits with distance ≤ 0.5 and discards weaker matches. Useful for reducing downstream LLM token cost when the top-`k` call returns loosely-related chunks. Appropriate values depend on the embedding model and distance strategy — inspect your actual scores before choosing one. Ignored (with a startup warning) under `VECTOR_DB_TYPE=atlas-mongo`, because Atlas returns a similarity score (higher = better) with inverted semantics.
- `RAG_UPLOAD_DIR`: (Optional) The directory where uploaded files are stored. Default value is "./uploads/".
- `PDF_EXTRACT_IMAGES`: (Optional) A boolean value indicating whether to extract images from PDF files. Default value is "False".
- `DEBUG_RAG_API`: (Optional) Set to "True" to show more verbose logging output in the server console, and to enable postgresql database routes
- `DEBUG_PGVECTOR_QUERIES`: (Optional) Set to "True" to enable detailed PostgreSQL query logging for pgvector operations. Useful for debugging performance issues with vector database queries.
- `CONSOLE_JSON`: (Optional) Set to "True" to log as json for Cloud Logging aggregations
- `EMBEDDINGS_PROVIDER`: (Optional) either "openai", "bedrock", "azure", "huggingface", "huggingfacetei", "google_genai", "vertexai", or "ollama", where "huggingface" uses sentence_transformers; defaults to "openai"
- `EMBEDDINGS_MODEL`: (Optional) Set a valid embeddings model to use from the configured provider.
    - **Defaults**
    - openai: "text-embedding-3-small"
    - azure: "text-embedding-3-small" (will be used as your Azure Deployment)
    - huggingface: "sentence-transformers/all-MiniLM-L6-v2"
    - huggingfacetei: "http://huggingfacetei:3000". Hugging Face TEI uses model defined on TEI service launch.
    - vertexai: "gemini-embedding-001"
    - ollama: "nomic-embed-text"
    - bedrock: "amazon.titan-embed-text-v1"
    - google_genai: "gemini-embedding-001"
- `EMBEDDINGS_CHUNK_SIZE`: (Optional) The chunk size used by the OpenAI and Azure embeddings clients to limit the number of inputs per request. Default value is `200`.
- `EMBEDDINGS_DIMENSIONS`: (Optional) Output vector size to request from the embedding model. Only honored by the `openai` and `azure` providers, and only supported by `text-embedding-3-*` models. Leave unset to use the model's native dimensionality (1536 for `text-embedding-3-small`, 3072 for `text-embedding-3-large`). Setting a smaller value (e.g. `512`, `1024`) trades some retrieval quality for lower storage cost and faster similarity search. Note: do not change this on an existing collection — all vectors in a `pgvector` column must share the same dimensionality.
- `RAG_AZURE_OPENAI_API_VERSION`: (Optional) Default is `2023-05-15`. The version of the Azure OpenAI API.
- `RAG_AZURE_OPENAI_API_KEY`: (Optional) The API key for Azure OpenAI service.
    - Note: `AZURE_OPENAI_API_KEY` will work but `RAG_AZURE_OPENAI_API_KEY` will override it in order to not conflict with LibreChat setting.
- `RAG_AZURE_OPENAI_ENDPOINT`: (Optional) The endpoint URL for Azure OpenAI service, including the resource.
    - Example: `https://YOUR_RESOURCE_NAME.openai.azure.com`.
    - Note: `AZURE_OPENAI_ENDPOINT` will work but `RAG_AZURE_OPENAI_ENDPOINT` will override it in order to not conflict with LibreChat setting.
- `HF_TOKEN`: (Optional) if needed for `huggingface` option.
- `OLLAMA_BASE_URL`: (Optional) defaults to `http://ollama:11434`.
- `ATLAS_SEARCH_INDEX`: (Optional) the name of the vector search index if using Atlas MongoDB, defaults to `vector_index`
- `MONGO_VECTOR_COLLECTION`: Deprecated for MongoDB, please use `ATLAS_SEARCH_INDEX` and `COLLECTION_NAME`
- `AWS_DEFAULT_REGION`: (Optional) defaults to `us-east-1`
- `AWS_ACCESS_KEY_ID`: (Optional) needed for bedrock embeddings
- `AWS_SECRET_ACCESS_KEY`: (Optional) needed for bedrock embeddings
- `GOOGLE_API_KEY`, `GOOGLE_KEY`, `RAG_GOOGLE_API_KEY`: (Optional) Google API key for Google GenAI embeddings. Priority order: RAG_GOOGLE_API_KEY > GOOGLE_KEY > GOOGLE_API_KEY
- `AWS_SESSION_TOKEN`: (Optional) may be needed for bedrock embeddings
- `GOOGLE_APPLICATION_CREDENTIALS`: (Optional) needed for Google VertexAI embeddings. This should be a path to a service account credential file in JSON format.
- `GOOGLE_CLOUD_PROJECT`: (Optional) Google Cloud project ID, needed for VertexAI embeddings.
- `GOOGLE_CLOUD_LOCATION`: (Optional) Google Cloud region for VertexAI embeddings. Defaults to `us-central1`.
- `RAG_CHECK_EMBEDDING_CTX_LENGTH` (Optional) Default is true, disabling this will send raw input to the embedder, use this for custom embedding models.

Make sure to set these environment variables before running the application. You can set them in a `.env` file or as system environment variables.

### Embedding Batch Processing

For large files, you can enable batched embedding processing to reduce memory consumption. This is particularly useful in memory-constrained environments like Kubernetes pods with memory limits.

#### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_BATCH_SIZE` | `500` | Number of document chunks to process per batch. `0` disables batching (original behavior). |
| `EMBEDDING_MAX_QUEUE_SIZE` | `3` | Maximum number of batches to buffer in memory during async processing. |
| `PARALLEL_EXECUTION` | `2` | Maximum number of async embedding/database insertion consumers per file when batching is enabled. |

#### Recommended Settings

For `text-embedding-3-small` model:
- `EMBEDDING_BATCH_SIZE=750` - Good balance of throughput and memory

For memory-constrained environments (< 2GB RAM):
- `EMBEDDING_BATCH_SIZE=100-250`

For high-throughput environments:
- `EMBEDDING_BATCH_SIZE=1000-2000`
- `EMBEDDING_MAX_QUEUE_SIZE=5`
- Increase `PARALLEL_EXECUTION` cautiously; it applies per active file upload.

#### Behavior

When `EMBEDDING_BATCH_SIZE > 0`:
- Documents are processed in batches of the specified size
- Up to `PARALLEL_EXECUTION` batches for the same file can be embedded and inserted concurrently
- Source documents are parsed and split incrementally, with at most `RAG_INGESTION_WINDOW_SIZE` prepared chunks retained before insertion
- Complete file ingestions are limited by `RAG_INGESTION_CONCURRENCY`; `PARALLEL_EXECUTION` still controls batch consumers within one active file
- On failure, remaining batch work is stopped and successfully inserted documents are rolled back
- Memory usage is bounded by the preparation window plus queued and active embedding batches
- Ingestion lifecycle logs include route, user, file, chunk count, file size, elapsed time, and selected process memory context. Per-batch queue/insert progress is logged at debug level

When `EMBEDDING_BATCH_SIZE <= 0`:
- All documents are processed at once (original behavior)
- Better for small files or memory-rich environments

### Use Atlas MongoDB as Vector Database

Instead of using the default pgvector, we could use [Atlas MongoDB](https://www.mongodb.com/products/platform/atlas-vector-search) as the vector database. To do so, set the following environment variables

```env
VECTOR_DB_TYPE=atlas-mongo
ATLAS_MONGO_DB_URI=<mongodb+srv://...>
COLLECTION_NAME=<vector collection>
ATLAS_SEARCH_INDEX=<vector search index>
```

The `ATLAS_MONGO_DB_URI` could be the same or different from what is used by LibreChat. Even if it is the same, the `$COLLECTION_NAME` collection needs to be a completely new one, separate from all collections used by LibreChat. In addition,  create a vector search index for collection above (remember to assign `$ATLAS_SEARCH_INDEX`) with the following json:

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "file_id",
      "type": "filter"
    },
    {
      "path": "user_id",
      "type": "filter"
    }
  ]
}
```

Follow one of the [four documented methods](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure) to create the vector index.

> **Upgrading an existing Atlas deployment:** `user_id` is a required filter field
> as of the release described under [Retrieval scope](#retrieval-scope). Retrieval
> now filters on it, and Atlas Vector Search rejects a `$vectorSearch` pre-filter on
> a path the index does not declare — so add it to the index definition **before**
> deploying, or `/query` and `/query_multiple` will start returning errors.

#### Create a `file_id` Index (recommended)

We recommend creating a standard MongoDB index on `file_id` to keep lookups fast. After creating the collection, run the following once (via Atlas UI, Compass, or `mongosh`):

```javascript
db.getCollection("<COLLECTION_NAME>").createIndex({ file_id: 1 })
```

Replace `<COLLECTION_NAME>` with the same collection used by the RAG API. This ensures lookups remain fast even as the number of embedded documents grows.


### Proxy Configuration

When using the RAG API with LibreChat and you need to configure proxy settings, you can set the `HTTP_PROXY` and `HTTPS_PROXY` environment variables in the [`docker-compose.override.yml`](https://www.librechat.ai/docs/configuration/docker_override) file (from the LibreChat repository):

```yaml
rag_api:
    environment:
        - HTTP_PROXY=<your-proxy>
        - HTTPS_PROXY=<your-proxy>
```

This configuration will ensure that all HTTP/HTTPS requests from the RAG API container are routed through your specified proxy server.


### Cloud Installation Settings:

#### AWS:
Make sure your RDS Postgres instance adheres to this requirement:

`The pgvector extension version 0.5.0 is available on database instances in Amazon RDS running PostgreSQL 15.4-R2 and higher, 14.9-R2 and higher, 13.12-R2 and higher, and 12.16-R2 and higher in all applicable AWS Regions, including the AWS GovCloud (US) Regions.`

In order to setup RDS Postgres with RAG API, you can follow these steps:

* Create a RDS Instance/Cluster using the provided [AWS Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_CreateDBInstance.html).
* Login to the RDS Cluster using the Endpoint connection string from the RDS Console or from your IaC Solution output.
* The login is via the *Master User*.
* Create a dedicated database for rag_api:
``` create database rag_api;```.
* Create a dedicated user\role for that database:
``` create role rag;```

* Switch to the database you just created: ```\c rag_api```
* Enable the Vector extension: ```create extension vector;```
* Use the documentation provided above to set up the connection string to the RDS Postgres Instance\Cluster.

Notes:
  * Even though you're logging with a Master user, it doesn't have all the super user privileges, that's why we cannot use the command: ```create role x with superuser;```
  * If you do not enable the extension, rag_api service will throw an error that it cannot create the extension due to the note above.

### Dev notes:

#### Running Tests

##### Prerequisites

Install test dependencies:

```bash
pip install -r test_requirements.txt
```

##### Running All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage (if pytest-cov is installed)
pytest --cov=app
```

##### Running Specific Test Files

```bash
# Run batch processing unit tests
pytest tests/test_batch_processing.py -v

# Run batch processing integration tests (memory optimization tests)
pytest tests/test_batch_processing_integration.py -v

# Run main API tests
pytest tests/test_main.py -v
```

##### Running Tests by Category

```bash
# Run only integration tests (marked with @pytest.mark.integration)
pytest -m integration -v

# Skip integration tests
pytest -m "not integration" -v

# Run only async tests
pytest -k "async" -v
```

##### Test Categories

| Test File | Description |
|-----------|-------------|
| `test_batch_processing.py` | Unit tests for batch processing functions |
| `test_batch_processing_integration.py` | Memory optimization and integration tests |
| `test_main.py` | API endpoint tests |
| `test_config.py` | Configuration tests |
| `test_middleware.py` | Middleware tests |
| `test_models.py` | Model tests |

##### Memory Optimization Tests

The `test_batch_processing_integration.py` file includes tests that verify the memory optimization behavior:

- **`test_memory_bounded_by_batch_size`**: Verifies that the number of documents in memory at any time is bounded by `EMBEDDING_BATCH_SIZE`
- **`test_memory_tracking_with_tracemalloc`**: Uses Python's `tracemalloc` to monitor memory usage during batch processing
- **`test_sync_memory_bounded_by_batch_size`**: Same verification for the synchronous code path

Run memory tests specifically:

```bash
pytest tests/test_batch_processing_integration.py::TestMemoryOptimization -v
pytest tests/test_batch_processing_integration.py::TestSyncBatchedMemory -v
```

#### Installing pre-commit formatter

Run the following commands to install pre-commit formatter, which uses [black](https://github.com/psf/black) code formatter:

```bash
pip install pre-commit
pre-commit install
```

