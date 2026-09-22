# app/routes/document_routes.py
import os
import sys
import uuid
from pathlib import Path
import hashlib
import traceback
import aiofiles
import aiofiles.os
import asyncio
import time
from shutil import copyfileobj
from typing import List, Iterable, Optional, Union, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    HTTPException,
    File,
    Form,
    Body,
    Query,
    status,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache

if TYPE_CHECKING:
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from app.services.vector_store.atlas_mongo_vector import AtlasMongoVector
    from langchain_community.vectorstores.pgvector import PGVector as PgVector

from app.config import (
    logger,
    vector_store,
    VECTOR_DB_TYPE,
    VectorDBType,
    RAG_UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_QUEUE_SIZE,
    PARALLEL_EXECUTION,
    RAG_DISTANCE_THRESHOLD,
)
from app.scope import file_clause, files_clause, resolve_scope

# Warn once at import time if the user set a threshold under Atlas, where
# the score direction is inverted (Atlas vectorSearchScore: higher = better)
# and naive `score <= threshold` would keep the *weaker* matches. We scope
# the filter to pgvector only until we grow a first-class "min similarity"
# semantic for Atlas.
#
# Inspect the raw env var here rather than the parsed RAG_DISTANCE_THRESHOLD:
# the parser in app.config deliberately skips the float() cast under Atlas
# (so non-numeric stale values don't break startup), which means the parsed
# value is always None for Atlas — and relying on it would suppress the
# warning we want operators to see.
if VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO and os.getenv(
    "RAG_DISTANCE_THRESHOLD"
) not in (None, ""):
    logger.warning(
        "RAG_DISTANCE_THRESHOLD is set but VECTOR_DB_TYPE=atlas-mongo; "
        "Atlas returns similarity scores (higher = better) which would "
        "invert the filter semantics, so the threshold will be ignored."
    )


def _apply_distance_threshold(documents):
    """Drop (doc, score) tuples whose distance exceeds RAG_DISTANCE_THRESHOLD.

    Only applied for pgvector, where similarity_search_with_score_by_vector
    returns a distance (lower = more similar). Skipped for Atlas because its
    score is a similarity (higher = better) and applying the same comparison
    would keep the weakest matches and drop the strongest.
    """
    if RAG_DISTANCE_THRESHOLD is None:
        return documents
    if VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
        return documents
    return [(doc, score) for doc, score in documents if score <= RAG_DISTANCE_THRESHOLD]


from app.constants import ERROR_MESSAGES
from app.models import (
    StoreDocument,
    QueryRequestBody,
    DocumentResponse,
    QueryMultipleBody,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.async_utils import run_in_executor
from app.utils.document_loader import (
    get_loader,
    clean_text,
    process_documents,
    cleanup_temp_encoding_file,
)
from app.utils.health import is_health_ok

router = APIRouter()

_INGESTION_ATTEMPT_ID_KEY = "_rag_ingestion_attempt_id"
_INGESTION_ATTEMPT_STARTED_AT_NS_KEY = "_rag_ingestion_attempt_started_at_ns"
_INGESTION_CHUNK_INDEX_KEY = "_rag_chunk_index"


def _tag_documents_for_ingestion(documents: List[Document], file_id: str) -> str:
    """Attach one attempt identity and source position to every document."""
    ingestion_attempt_id = uuid.uuid4().hex
    ingestion_attempt_started_at_ns = time.time_ns()

    for chunk_index, document in enumerate(documents):
        document.metadata = {
            **(document.metadata or {}),
            "file_id": file_id,
            _INGESTION_CHUNK_INDEX_KEY: chunk_index,
            _INGESTION_ATTEMPT_ID_KEY: ingestion_attempt_id,
            _INGESTION_ATTEMPT_STARTED_AT_NS_KEY: ingestion_attempt_started_at_ns,
        }

    return ingestion_attempt_id


def _order_documents_by_chunk_index(documents: List[Document]) -> List[Document]:
    """Group ingestion attempts and restore source order within each group.

    Repeated ingestions intentionally retain every row currently returned by the
    vector store. Legacy rows stay first in their received order, followed by marked
    attempts in start-time order. Within a group, preserve received order instead of
    guessing when a chunk index is missing, invalid, or duplicated.
    """

    def order_group(group: List[Document]) -> List[Document]:
        received_group = list(group)
        chunk_indexes = []

        for document in received_group:
            chunk_index = (document.metadata or {}).get(_INGESTION_CHUNK_INDEX_KEY)
            if (
                isinstance(chunk_index, bool)
                or not isinstance(chunk_index, int)
                or chunk_index < 0
            ):
                return received_group
            chunk_indexes.append(chunk_index)

        if len(set(chunk_indexes)) != len(chunk_indexes):
            return received_group

        return [
            document
            for _, document in sorted(
                zip(chunk_indexes, received_group), key=lambda item: item[0]
            )
        ]

    attempt_groups = {}
    legacy_documents = []
    for document in documents:
        metadata = document.metadata or {}
        attempt_id = metadata.get(_INGESTION_ATTEMPT_ID_KEY)
        started_at_ns = metadata.get(_INGESTION_ATTEMPT_STARTED_AT_NS_KEY)
        if (
            not isinstance(attempt_id, str)
            or not attempt_id
            or isinstance(started_at_ns, bool)
            or not isinstance(started_at_ns, int)
            or started_at_ns < 0
        ):
            legacy_documents.append(document)
            continue

        attempt_groups.setdefault((started_at_ns, attempt_id), []).append(document)

    if not attempt_groups:
        return legacy_documents

    ordered_documents = list(legacy_documents)
    for attempt_key in sorted(attempt_groups):
        ordered_documents.extend(order_group(attempt_groups[attempt_key]))
    return ordered_documents


def calculate_num_batches(total: int, batch_size: int) -> int:
    """Calculate the number of batches needed to process total items."""
    if batch_size <= 0:
        return 1
    return (total + batch_size - 1) // batch_size


def get_user_id(request: Request, entity_id: str = None) -> str:
    """Extract user ID from request or entity_id."""
    if not hasattr(request.state, "user"):
        return entity_id if entity_id else "public"
    else:
        return entity_id if entity_id else request.state.user.get("id")


def get_process_memory_details() -> str:
    """Return lightweight process memory details for logging."""
    parts = []

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    rss_kib = int(line.split()[1])
                    parts.append(f"rss_mb={rss_kib / 1024:.1f}")
                elif line.startswith("VmHWM:"):
                    hwm_kib = int(line.split()[1])
                    parts.append(f"rss_peak_mb={hwm_kib / 1024:.1f}")
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass

    try:
        import resource

        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Darwin reports ru_maxrss in bytes; Linux reports it in KiB.
        max_rss_mb = (
            max_rss / (1024 * 1024) if sys.platform == "darwin" else max_rss / 1024
        )

        if not any(part.startswith("rss_peak_mb=") for part in parts):
            parts.append(f"rss_peak_mb={max_rss_mb:.1f}")
    except (ImportError, AttributeError, OSError, ValueError):
        pass

    return " | ".join(parts) if parts else "rss_mb=unknown"


def build_ingestion_context(
    route_name: str,
    user_id: str,
    file_id: str,
    filename: str,
    content_type: Optional[str] = None,
    temp_file_path: Optional[str] = None,
    known_type: Optional[str] = None,
    file_ext: Optional[str] = None,
    chunk_count: Optional[int] = None,
    include_memory: bool = False,
) -> str:
    """Build a compact ingestion context string for logs."""
    parts = [
        f"route={route_name}",
        f"user_id={user_id}",
        f"file_id={file_id}",
        f"filename={filename}",
    ]
    if content_type:
        parts.append(f"content_type={content_type}")
    if temp_file_path:
        try:
            file_size_bytes = os.path.getsize(temp_file_path)
            parts.append(f"file_size_bytes={file_size_bytes}")
        except OSError:
            pass
    if known_type:
        parts.append(f"known_type={known_type}")
    if file_ext:
        parts.append(f"file_ext={file_ext}")
    if chunk_count is not None:
        parts.append(f"chunk_count={chunk_count}")
    if include_memory:
        parts.append(get_process_memory_details())
    return " | ".join(parts)


async def save_upload_file_async(file: UploadFile, temp_file_path: str) -> None:
    """Save uploaded file asynchronously."""
    try:
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            chunk_size = 64 * 1024  # 64 KB
            while content := await file.read(chunk_size):
                await temp_file.write(content)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )


def save_upload_file_sync(file: UploadFile, temp_file_path: str) -> None:
    """Save uploaded file synchronously."""
    try:
        with open(temp_file_path, "wb") as temp_file:
            copyfileobj(file.file, temp_file)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )


def validate_file_path(base_dir: str, file_path: str) -> Optional[str]:
    """Validate that file_path resolves within base_dir. Returns resolved absolute path or None."""
    if not file_path or not file_path.strip():
        return None
    try:
        allowed = Path(base_dir).resolve()
        requested = Path(os.path.join(base_dir, file_path)).resolve()
        requested.relative_to(allowed)
        return str(requested)
    except (ValueError, RuntimeError, TypeError, OSError):
        return None


def _make_unique_temp_path(user_id: str, filename: str) -> Optional[str]:
    """Build a unique temp file path under RAG_UPLOAD_DIR/{user_id}/ to prevent
    concurrent upload collisions. Returns a validated absolute path, or None if
    the raw filename would escape RAG_UPLOAD_DIR (path traversal rejection)."""
    # Validate the raw filename to reject traversal attempts
    if validate_file_path(RAG_UPLOAD_DIR, os.path.join(user_id, filename)) is None:
        return None
    # unique_name is stem + "_" + [0-9a-f]{32} + suffix — no path separators,
    # so it cannot escape the directory validated above.
    p = Path(filename)
    unique_name = f"{p.stem}_{uuid.uuid4().hex}{p.suffix}"
    return str(Path(RAG_UPLOAD_DIR, user_id, unique_name).resolve())


async def load_file_content(
    filename: str,
    content_type: str,
    file_path: str,
    executor,
    raw_text: bool = False,
) -> tuple:
    """Load file content using appropriate loader.

    Pass ``raw_text=True`` when the caller wants verbatim file contents (e.g.
    the ``/text`` endpoint) so text-formatted files are not semantically
    parsed.
    """
    loader = None
    try:
        loader, known_type, file_ext = get_loader(
            filename, content_type, file_path, raw_text=raw_text
        )
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(executor, lambda: list(loader.lazy_load()))
        return data, known_type, file_ext
    finally:
        # Clean up temporary UTF-8 file if it was created for encoding conversion
        if loader is not None:
            cleanup_temp_encoding_file(loader)


def extract_text_from_documents(documents: List[Document], file_ext: str) -> str:
    """Extract text content from loaded documents."""
    text_content = ""
    if documents:
        for doc in documents:
            if hasattr(doc, "page_content"):
                # Clean text if it's a PDF
                if file_ext == "pdf":
                    text_content += clean_text(doc.page_content) + "\n"
                else:
                    text_content += doc.page_content + "\n"

    # Remove trailing newline
    return text_content.rstrip("\n")


async def cleanup_temp_file_async(file_path: str) -> None:
    """Clean up temporary file asynchronously."""
    try:
        await aiofiles.os.remove(file_path)
    except Exception as e:
        logger.error(
            "Failed to remove temporary file | Path: %s | Error: %s | Traceback: %s",
            file_path,
            str(e),
            traceback.format_exc(),
        )


@router.get("/ids")
async def get_all_ids(request: Request, entity_id: str = None):
    scope = resolve_scope(request, entity_id)
    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.get_all_ids(
                owners=scope.owners, executor=request.app.state.thread_pool
            )
        else:
            ids = vector_store.get_all_ids(owners=scope.owners)

        return list(set(ids))
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_all_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to get all IDs | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    try:
        if await is_health_ok():
            return {"status": "UP"}
        else:
            logger.error("Health check failed")
            return {"status": "DOWN"}, 503
    except Exception as e:
        logger.error(
            "Error during health check | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        return {"status": "DOWN", "error": str(e)}, 503


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(
    request: Request, ids: list[str] = Query(...), entity_id: str = None
):
    scope = resolve_scope(request, entity_id)
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                ids, owners=scope.owners, executor=request.app.state.thread_pool
            )
            documents = await vector_store.get_documents_by_ids(
                ids, owners=scope.owners, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(ids, owners=scope.owners)
            documents = vector_store.get_documents_by_ids(ids, owners=scope.owners)

        # A file outside the caller's scope reads as absent rather than refused,
        # so this route is not an existence oracle over the deployment.
        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given IDs"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_documents_by_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error getting documents by IDs | IDs: %s | Error: %s | Traceback: %s",
            ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_documents(
    request: Request, document_ids: List[str] = Body(...), entity_id: str = None
):
    scope = resolve_scope(request, entity_id)
    try:
        # Resolve existence within the caller's scope *before* deleting anything.
        # A request mixing owned and unknown ids would otherwise destroy the owned
        # rows and still answer 404, leaving the caller to believe nothing went.
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                document_ids,
                owners=scope.owners,
                executor=request.app.state.thread_pool,
            )
        else:
            existing_ids = vector_store.get_filtered_ids(
                document_ids, owners=scope.owners
            )

        if not all(id in existing_ids for id in document_ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        if isinstance(vector_store, AsyncPgVector):
            await vector_store.delete_scoped(
                ids=document_ids,
                owners=scope.owners,
                executor=request.app.state.thread_pool,
            )
        else:
            vector_store.delete_scoped(ids=document_ids, owners=scope.owners)

        file_count = len(document_ids)
        return {
            "message": f"Documents for {file_count} file{'s' if file_count > 1 else ''} deleted successfully"
        }
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in delete_documents | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to delete documents | IDs: %s | Error: %s | Traceback: %s",
            document_ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


# Cache query vectors only, never scoped search results. Both cache hits and
# misses run in the request worker pool so a miss cannot block the event loop.
@lru_cache(maxsize=128)
def get_cached_query_embedding(query: str):
    return vector_store.embedding_function.embed_query(query)


@router.post("/query")
async def query_embeddings_by_file_id(
    body: QueryRequestBody,
    request: Request,
):
    scope = resolve_scope(request, body.entity_id)

    try:
        embedding = await run_in_executor(
            request.app.state.thread_pool, get_cached_query_embedding, body.query
        )
        query_filter = scope.predicate(file_clause(body.file_id))

        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter=query_filter,
                executor=request.app.state.thread_pool,
            )
        else:
            documents = await run_in_executor(
                request.app.state.thread_pool,
                vector_store.similarity_search_with_score_by_vector,
                embedding,
                k=body.k,
                filter=query_filter,
            )

        return _apply_distance_threshold(documents)

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_id | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query embeddings | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


async def _process_documents_async_pipeline(
    documents: List[Document],
    file_id: str,
    vector_store: "AsyncPgVector",
    executor: "ThreadPoolExecutor",
    parallel_execution: int = 1,
    user_id: str = "",
) -> List[str]:
    """
    Process documents using async producer-consumer pattern for batched embedding and insertion.

    Args:
        documents: List of Document objects to process
        file_id: Unique identifier for the file being processed
        vector_store: AsyncPgVector instance for document storage
        executor: ThreadPoolExecutor for concurrent operations

    Returns:
        List of document IDs that were successfully inserted
    """
    total_chunks = len(documents)
    if total_chunks == 0:
        return []

    ingestion_attempt_id = _tag_documents_for_ingestion(documents, file_id)

    # Create queues for producer-consumer pattern
    # embedding_queue is bounded to limit document data held in memory.
    embedding_queue = asyncio.Queue(maxsize=EMBEDDING_MAX_QUEUE_SIZE)
    all_ids = []
    successful_batch_ids = {}
    failure_event = asyncio.Event()
    in_flight_insert_tasks = set()
    cleanup_needed = False

    num_batches = calculate_num_batches(total_chunks, EMBEDDING_BATCH_SIZE)
    consumer_count = max(1, min(parallel_execution, num_batches))

    logger.info(
        "Starting async pipeline | user_id=%s | file_id=%s | total_chunks=%d | batch_size=%d | consumers=%d | %s",
        user_id,
        file_id,
        total_chunks,
        EMBEDDING_BATCH_SIZE,
        consumer_count,
        get_process_memory_details(),
    )

    async def batch_producer():
        """Produce document batches and put them in the queue."""
        try:
            for batch_idx in range(num_batches):
                if failure_event.is_set():
                    break

                start_idx = batch_idx * EMBEDDING_BATCH_SIZE
                end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, total_chunks)
                batch_documents = documents[start_idx:end_idx]
                batch_ids = [file_id] * len(batch_documents)

                logger.debug(
                    "Queueing batch | user_id=%s | file_id=%s | batch=%d/%d | chunk_range=%d-%d | batch_chunks=%d",
                    user_id,
                    file_id,
                    batch_idx + 1,
                    num_batches,
                    start_idx,
                    end_idx - 1,
                    len(batch_documents),
                )

                # Put batch in queue for processing
                await embedding_queue.put(
                    (batch_documents, batch_ids, batch_idx + 1, num_batches)
                )
            # Signal normal completion. Failure and cancellation paths are
            # cancelled by the caller and must never block enqueueing sentinels.
            if not failure_event.is_set():
                for _ in range(consumer_count):
                    await embedding_queue.put(None)
        except asyncio.CancelledError:
            failure_event.set()
            raise
        except Exception as e:
            failure_event.set()
            logger.error("Error in batch producer: %s", e)
            raise

    async def insert_batch(
        batch_documents: List[Document], batch_ids: List[str]
    ) -> List[str]:
        """Track started inserts so rollback waits for executor-backed work."""
        nonlocal cleanup_needed
        insert_task = asyncio.create_task(
            vector_store.aadd_documents(
                batch_documents, ids=batch_ids, executor=executor
            )
        )
        in_flight_insert_tasks.add(insert_task)
        try:
            batch_result_ids = await asyncio.shield(insert_task)
            cleanup_needed = True
            return batch_result_ids
        finally:
            if insert_task.done():
                try:
                    insert_task.result()
                    cleanup_needed = True
                except BaseException:
                    pass
                in_flight_insert_tasks.discard(insert_task)

    async def wait_for_in_flight_inserts() -> None:
        """Wait for started inserts before rollback can delete inserted vectors."""
        nonlocal cleanup_needed
        if not in_flight_insert_tasks:
            return

        pending_tasks = list(in_flight_insert_tasks)
        logger.warning(
            "Waiting for in-flight inserts before rollback | user_id=%s | file_id=%s | inserts=%d | %s",
            user_id,
            file_id,
            len(pending_tasks),
            get_process_memory_details(),
        )
        results = await asyncio.gather(*pending_tasks, return_exceptions=True)
        for insert_task, result in zip(pending_tasks, results):
            in_flight_insert_tasks.discard(insert_task)
            if not isinstance(result, BaseException):
                cleanup_needed = True

    async def embedding_consumer():
        """Consume batches from queue, embed and insert into database."""
        try:
            while True:
                item = await embedding_queue.get()
                try:
                    if item is None:  # End signal
                        break

                    # A peer can fail before gather() resumes and cancels this task.
                    # Acknowledge already-queued work without starting another insert.
                    if failure_event.is_set():
                        continue

                    batch_documents, batch_ids, batch_num, total_batches = item

                    logger.debug(
                        "Inserting batch | user_id=%s | file_id=%s | batch=%d/%d | batch_chunks=%d",
                        user_id,
                        file_id,
                        batch_num,
                        total_batches,
                        len(batch_documents),
                    )

                    try:
                        # Insert batch into database
                        batch_result_ids = await insert_batch(
                            batch_documents, batch_ids
                        )
                        successful_batch_ids[batch_num] = batch_result_ids
                    except Exception as e:
                        failure_event.set()
                        logger.error(
                            "Error processing batch | user_id=%s | file_id=%s | batch=%d/%d | error=%s | %s",
                            user_id,
                            file_id,
                            batch_num,
                            total_batches,
                            e,
                            get_process_memory_details(),
                        )
                        raise
                finally:
                    embedding_queue.task_done()

        except Exception as e:
            if not failure_event.is_set():
                failure_event.set()
                logger.error("Fatal error in embedding consumer: %s", e)
            raise

    producer_task = None
    consumer_tasks: List[asyncio.Task] = []

    try:
        # Start producer and consumers concurrently
        producer_task = asyncio.create_task(batch_producer())
        consumer_tasks = [
            asyncio.create_task(embedding_consumer()) for _ in range(consumer_count)
        ]

        # Wait for all tasks to complete
        await asyncio.gather(producer_task, *consumer_tasks, return_exceptions=False)

        for batch_num in range(1, num_batches + 1):
            all_ids.extend(successful_batch_ids[batch_num])

        logger.info(
            "Async pipeline completed | user_id=%s | file_id=%s | inserted_ids=%d | %s",
            user_id,
            file_id,
            len(all_ids),
            get_process_memory_details(),
        )

        return all_ids

    except (Exception, asyncio.CancelledError) as e:
        failure_event.set()
        logger.error(
            "Pipeline failed | user_id=%s | file_id=%s | inserted_batches=%d | error=%s | %s",
            user_id,
            file_id,
            len(successful_batch_ids),
            e,
            get_process_memory_details(),
        )
        if producer_task is not None or consumer_tasks:
            # if one of the tasks is still running, cancel it
            if producer_task is not None and not producer_task.done():
                producer_task.cancel()
            for consumer_task in consumer_tasks:
                if not consumer_task.done():
                    consumer_task.cancel()

            # Await cancelled tasks to ensure proper cleanup
            tasks_to_await = []
            if producer_task is not None:
                tasks_to_await.append(producer_task)
            tasks_to_await.extend(consumer_tasks)
            if tasks_to_await:
                await asyncio.gather(*tasks_to_await, return_exceptions=True)

        await wait_for_in_flight_inserts()

        if cleanup_needed:
            try:
                logger.warning(
                    "Performing rollback | user_id=%s | file_id=%s | %s",
                    user_id,
                    file_id,
                    get_process_memory_details(),
                )
                await vector_store.delete_by_metadata(
                    {
                        "file_id": file_id,
                        _INGESTION_ATTEMPT_ID_KEY: ingestion_attempt_id,
                    },
                    executor=executor,
                )
                logger.info(
                    "Rollback completed | user_id=%s | file_id=%s | %s",
                    user_id,
                    file_id,
                    get_process_memory_details(),
                )
            except Exception as cleanup_error:
                logger.error(
                    "Rollback failed | user_id=%s | file_id=%s | error=%s | %s",
                    user_id,
                    file_id,
                    cleanup_error,
                    get_process_memory_details(),
                )

        # Re-raise the original error
        raise


async def _process_documents_batched_sync(
    documents: List[Document],
    file_id: str,
    user_id: str,
    vector_store: Union["PgVector", "AtlasMongoVector"],
    executor: "ThreadPoolExecutor",
) -> List[str]:
    """
    Process documents in batches using synchronous vector store operations.

    Args:
        documents: List of Document objects to process
        file_id: Unique identifier for the file being processed
        user_id: Owner of the chunks being written; scopes the rollback
        vector_store: Synchronous vector store instance (ExtendedPgVector or AtlasMongoVector)
        executor: ThreadPoolExecutor for running sync operations

    Returns:
        List of document IDs that were successfully inserted
    """
    total_chunks = len(documents)
    if total_chunks == 0:
        return []

    all_ids = []
    num_batches = calculate_num_batches(total_chunks, EMBEDDING_BATCH_SIZE)

    logger.info(
        "Processing file %s with sync batching: %d batches of %d chunks each",
        file_id,
        num_batches,
        EMBEDDING_BATCH_SIZE,
    )

    loop = asyncio.get_running_loop()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * EMBEDDING_BATCH_SIZE
        end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, total_chunks)
        batch_documents = documents[start_idx:end_idx]
        batch_ids = [file_id] * len(batch_documents)

        logger.info(
            "Processing batch %d/%d: chunks %d-%d (%d chunks)",
            batch_idx + 1,
            num_batches,
            start_idx,
            end_idx - 1,
            len(batch_documents),
        )

        try:
            # Wrap sync call in executor to avoid blocking the event loop
            batch_result_ids = await loop.run_in_executor(
                executor,
                lambda docs=batch_documents, ids=batch_ids: vector_store.add_documents(
                    docs, ids=ids
                ),
            )
            all_ids.extend(batch_result_ids)

        except Exception as batch_error:
            logger.error("Batch %d failed: %s", batch_idx + 1, batch_error)

            # Rollback entire file from vector store
            if (
                all_ids
            ):  # any batch succeeded (i.e., any chunks for this file were inserted)
                logger.warning("Rolling back file %s due to batch failure", file_id)
                try:
                    await loop.run_in_executor(
                        executor,
                        lambda: vector_store.delete_scoped(
                            ids=[file_id], owners=[user_id]
                        ),
                    )
                    logger.info("Rollback completed for file %s", file_id)
                except Exception as rollback_error:
                    logger.error(
                        "Rollback failed for file %s: %s", file_id, rollback_error
                    )

            raise batch_error

    return all_ids


def generate_digest(page_content: str) -> str:
    return hashlib.md5(page_content.encode("utf-8", "ignore")).hexdigest()


def _prepare_documents_sync(
    data: Iterable[Document],
    file_id: str,
    user_id: str,
    clean_content: bool,
) -> List[Document]:
    """
    Synchronous document preparation - runs in executor to avoid blocking event loop.
    Handles text splitting, cleaning, and metadata preparation.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(data)

    # If `clean_content` is True, clean the page_content of each document (remove null bytes)
    if clean_content:
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

    # Preparing documents with page content and metadata for insertion.
    return [
        Document(
            page_content=doc.page_content,
            metadata={
                "file_id": file_id,
                "user_id": user_id,
                "digest": generate_digest(doc.page_content),
                **(doc.metadata or {}),
            },
        )
        for doc in documents
    ]


async def store_data_in_vector_db(
    data: Iterable[Document],
    file_id: str,
    user_id: str = "",
    clean_content: bool = False,
    executor=None,
    route_name: str = "unknown",
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    temp_file_path: Optional[str] = None,
) -> bool:
    start_time = time.perf_counter()
    # Run document preparation in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(
        executor,
        _prepare_documents_sync,
        data,
        file_id,
        user_id,
        clean_content,
    )

    logger.info(
        "Documents prepared | %s",
        build_ingestion_context(
            route_name=route_name,
            user_id=user_id,
            file_id=file_id,
            filename=filename or file_id,
            content_type=content_type,
            temp_file_path=temp_file_path,
            chunk_count=len(docs),
            include_memory=True,
        ),
    )

    try:
        if EMBEDDING_BATCH_SIZE <= 0:
            # synchronously embed the file and insert into vector store in one go
            if isinstance(vector_store, AsyncPgVector):
                _tag_documents_for_ingestion(docs, file_id)
                ids = await vector_store.aadd_documents(
                    docs, ids=[file_id] * len(docs), executor=executor
                )
            else:
                ids = vector_store.add_documents(docs, ids=[file_id] * len(docs))
        else:
            # asynchronously embed the file and insert into vector store as it is embedding
            # to lessen memory impact and speed up slightly as the majority of the document
            # is inserted into db by the time it is fully embedded

            if isinstance(vector_store, AsyncPgVector):
                ids = await _process_documents_async_pipeline(
                    docs,
                    file_id,
                    vector_store,
                    executor,
                    parallel_execution=max(1, PARALLEL_EXECUTION),
                    user_id=user_id,
                )
            else:
                # Fallback to batched processing for sync vector stores
                ids = await _process_documents_batched_sync(
                    docs, file_id, user_id, vector_store, executor
                )

        logger.info(
            "Ingestion completed | %s | inserted_ids=%d | elapsed_ms=%d",
            build_ingestion_context(
                route_name=route_name,
                user_id=user_id,
                file_id=file_id,
                filename=filename or file_id,
                content_type=content_type,
                temp_file_path=temp_file_path,
                chunk_count=len(docs),
                include_memory=True,
            ),
            len(ids),
            int((time.perf_counter() - start_time) * 1000),
        )
        return {"message": "Documents added successfully", "ids": ids}

    except Exception as e:
        logger.error(
            "Failed to store data in vector DB | %s | elapsed_ms=%d | Error: %s | Traceback: %s",
            build_ingestion_context(
                route_name=route_name,
                user_id=user_id,
                file_id=file_id,
                filename=filename or file_id,
                content_type=content_type,
                temp_file_path=temp_file_path,
                chunk_count=len(docs),
                include_memory=True,
            ),
            int((time.perf_counter() - start_time) * 1000),
            str(e),
            traceback.format_exc(),
        )
        return {"message": "An error occurred while adding documents.", "error": str(e)}


@router.post("/local/embed")
async def embed_local_file(
    document: StoreDocument, request: Request, entity_id: str = None
):
    user_id = get_user_id(request, entity_id)
    file_path = validate_file_path(RAG_UPLOAD_DIR, document.filepath)

    # Check if the file exists and if it is within the allowed upload directory
    if file_path is None or not os.path.exists(file_path):
        logger.warning(
            "Path validation failed for local embed | route=local_embed | user_id=%s | file_id=%s | filename=%s | requested_path=%s",
            user_id,
            document.file_id,
            document.filename,
            document.filepath,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.FILE_NOT_FOUND,
        )

    loader = None
    try:
        logger.info(
            "Ingestion started | %s",
            build_ingestion_context(
                route_name="local_embed",
                user_id=user_id,
                file_id=document.file_id,
                filename=document.filename,
                content_type=document.file_content_type,
                temp_file_path=file_path,
            ),
        )
        loader, known_type, file_ext = get_loader(
            document.filename, document.file_content_type, file_path
        )
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            request.app.state.thread_pool, lambda: list(loader.lazy_load())
        )

        result = await store_data_in_vector_db(
            data,
            document.file_id,
            user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
            route_name="local_embed",
            filename=document.filename,
            content_type=document.file_content_type,
            temp_file_path=file_path,
        )

        if result:
            return {
                "status": True,
                "file_id": document.file_id,
                "filename": document.filename,
                "known_type": known_type,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERROR_MESSAGES.DEFAULT(),
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_local_file | route=local_embed | user_id=%s | file_id=%s | filename=%s | status=%d | detail=%s",
            user_id,
            document.file_id,
            document.filename,
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Unhandled exception in embed_local_file | route=local_embed | user_id=%s | file_id=%s | filename=%s | error=%s | traceback=%s",
            user_id,
            document.file_id,
            document.filename,
            str(e),
            traceback.format_exc(),
        )
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )
    finally:
        # Clean up temporary UTF-8 file if it was created for encoding conversion
        if loader is not None:
            cleanup_temp_encoding_file(loader)


@router.post("/embed")
async def embed_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    response_status = True
    response_message = "File processed successfully."
    known_type = None

    user_id = get_user_id(request, entity_id)
    validated_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_file_path is None:
        logger.warning(
            "Path validation failed for embed | route=embed | user_id=%s | file_id=%s | filename=%s",
            user_id,
            file_id,
            file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        logger.info(
            "Ingestion started | %s",
            build_ingestion_context(
                route_name="embed",
                user_id=user_id,
                file_id=file_id,
                filename=file.filename,
                content_type=file.content_type,
                temp_file_path=validated_file_path,
            ),
        )
        os.makedirs(os.path.dirname(validated_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_file_path)
        data, known_type, file_ext = await load_file_content(
            file.filename,
            file.content_type,
            validated_file_path,
            request.app.state.thread_pool,
        )

        logger.debug(
            "Loading file | filename=%s | content_type=%s | file_ext=%s | known_type=%s",
            file.filename,
            file.content_type,
            file_ext,
            known_type,
        )

        result = await store_data_in_vector_db(
            data=data,
            file_id=file_id,
            user_id=user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
            route_name="embed",
            filename=file.filename,
            content_type=file.content_type,
            temp_file_path=validated_file_path,
        )

        if not result:
            response_status = False
            response_message = "Failed to process/store the file data."
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
        elif "error" in result:
            response_status = False
            response_message = "Failed to process/store the file data."
            if isinstance(result["error"], str):
                response_message = result["error"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An unspecified error occurred.",
                )
    except HTTPException as http_exc:
        response_status = False
        response_message = f"HTTP Exception: {http_exc.detail}"
        logger.error(
            "HTTP Exception in embed_file | route=embed | user_id=%s | file_id=%s | filename=%s | status=%d | detail=%s",
            user_id,
            file_id,
            file.filename,
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        response_status = False
        response_message = f"Error during file processing: {str(e)}"
        logger.error(
            "Error during file processing | route=embed | user_id=%s | file_id=%s | filename=%s | error=%s | traceback=%s",
            user_id,
            file_id,
            file.filename,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        await cleanup_temp_file_async(validated_file_path)

    return {
        "status": response_status,
        "message": response_message,
        "file_id": file_id,
        "filename": file.filename,
        "known_type": known_type,
    }


@router.get("/documents/{id}/context")
async def load_document_context(request: Request, id: str, entity_id: str = None):
    ids = [id]
    scope = resolve_scope(request, entity_id)
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                ids, owners=scope.owners, executor=request.app.state.thread_pool
            )
            documents = await vector_store.get_documents_by_ids(
                ids, owners=scope.owners, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(ids, owners=scope.owners)
            documents = vector_store.get_documents_by_ids(ids, owners=scope.owners)

        # A file outside the caller's scope reads as absent rather than refused,
        # so this route is not an existence oracle over the deployment.
        if not all(id in existing_ids for id in ids):
            raise HTTPException(
                status_code=404, detail="The specified file_id was not found"
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No document found for the given ID"
            )

        return process_documents(_order_documents_by_chunk_index(documents))
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in load_document_context | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error loading document context | Document ID: %s | Error: %s | Traceback: %s",
            id,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.post("/embed-upload")
async def embed_file_upload(
    request: Request,
    file_id: str = Form(...),
    uploaded_file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    user_id = get_user_id(request, entity_id)

    validated_temp_file_path = _make_unique_temp_path(user_id, uploaded_file.filename)

    if validated_temp_file_path is None:
        logger.warning(
            "Path validation failed for embed-upload | route=embed_upload | user_id=%s | file_id=%s | filename=%s",
            user_id,
            file_id,
            uploaded_file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        logger.info(
            "Ingestion started | %s",
            build_ingestion_context(
                route_name="embed_upload",
                user_id=user_id,
                file_id=file_id,
                filename=uploaded_file.filename,
                content_type=uploaded_file.content_type,
                temp_file_path=validated_temp_file_path,
            ),
        )
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(uploaded_file, validated_temp_file_path)
        data, known_type, file_ext = await load_file_content(
            uploaded_file.filename,
            uploaded_file.content_type,
            validated_temp_file_path,
            request.app.state.thread_pool,
        )

        result = await store_data_in_vector_db(
            data,
            file_id,
            user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
            route_name="embed_upload",
            filename=uploaded_file.filename,
            content_type=uploaded_file.content_type,
            temp_file_path=validated_temp_file_path,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_file_upload | route=embed_upload | user_id=%s | file_id=%s | filename=%s | status=%d | detail=%s",
            user_id,
            file_id,
            uploaded_file.filename,
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error during file processing | route=embed_upload | user_id=%s | file_id=%s | filename=%s | error=%s | traceback=%s",
            user_id,
            file_id,
            uploaded_file.filename,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        await cleanup_temp_file_async(validated_temp_file_path)

    return {
        "status": True,
        "message": "File processed successfully.",
        "file_id": file_id,
        "filename": uploaded_file.filename,
        "known_type": known_type,
    }


@router.post("/query_multiple")
async def query_embeddings_by_file_ids(request: Request, body: QueryMultipleBody):
    scope = resolve_scope(request, body.entity_id)
    try:
        # Get the embedding of the query text
        embedding = await run_in_executor(
            request.app.state.thread_pool, get_cached_query_embedding, body.query
        )
        query_filter = scope.predicate(files_clause(body.file_ids))

        # Perform similarity search with the query embedding and filter by the file_ids in metadata
        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter=query_filter,
                executor=request.app.state.thread_pool,
            )
        else:
            documents = await run_in_executor(
                request.app.state.thread_pool,
                vector_store.similarity_search_with_score_by_vector,
                embedding,
                k=body.k,
                filter=query_filter,
            )

        documents = _apply_distance_threshold(documents)

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given query"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query multiple embeddings | File IDs: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_ids,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text")
async def extract_text_from_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    """
    Extract text content from an uploaded file without creating embeddings.
    Returns the raw text content for text parsing purposes.
    """
    user_id = get_user_id(request, entity_id)
    validated_temp_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_temp_file_path is None:
        logger.warning("Path validation failed for text extraction: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_temp_file_path)
        data, known_type, file_ext = await load_file_content(
            file.filename,
            file.content_type,
            validated_temp_file_path,
            request.app.state.thread_pool,
            raw_text=True,
        )

        # Extract text content from loaded documents
        text_content = extract_text_from_documents(data, file_ext)

        return {
            "text": text_content,
            "file_id": file_id,
            "filename": file.filename,
            "known_type": known_type,
        }

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in extract_text_from_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error during text extraction | File: %s | Error: %s | Traceback: %s",
            file.filename,
            str(e),
            traceback.format_exc(),
        )
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error during text extraction: {str(e)}",
            )
    finally:
        await cleanup_temp_file_async(validated_temp_file_path)
