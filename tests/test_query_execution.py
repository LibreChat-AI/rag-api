"""Exercise query dispatch with real worker threads and no external services."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

from app.config import VectorDBType
from app.models import QueryMultipleBody, QueryRequestBody
from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.services.vector_store.extended_pg_vector import ExtendedPgVector


class DummyAsyncPgVector(AsyncPgVector):
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self._thread_pool = None
        self._bind = None


@pytest.fixture(params=["single", "multiple"])
def query_route(request):
    return request.param


@pytest.fixture(params=["async", "sync"])
def query_context(request, monkeypatch, query_route):
    """Use the real async adapter or the route's synchronous fallback."""
    embedding = Mock(return_value=[0.1, 0.2, 0.3])
    provider = SimpleNamespace(embed_query=embedding)
    hits = [
        (Document(page_content="near", metadata={"file_id": "file-1"}), 0.2),
        (Document(page_content="far", metadata={"file_id": "file-1"}), 0.8),
    ]
    search = Mock(return_value=hits)
    if request.param == "async":
        store = DummyAsyncPgVector(provider)
        monkeypatch.setattr(
            ExtendedPgVector,
            "similarity_search_with_score_by_vector",
            lambda self, *args, **kwargs: search(*args, **kwargs),
        )
    else:
        store = SimpleNamespace(
            embedding_function=provider,
            similarity_search_with_score_by_vector=search,
        )
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "RAG_DISTANCE_THRESHOLD", None)
    monkeypatch.setattr(document_routes, "VECTOR_DB_TYPE", VectorDBType.PGVECTOR)
    document_routes.get_cached_query_embedding.cache_clear()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="query-test") as pool:
        req = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(thread_pool=pool)),
            state=SimpleNamespace(user={"id": "owner-1"}),
        )

        async def query(text="question", entity_id=None):
            body_args = dict(query=text, k=7, entity_id=entity_id)
            if query_route == "single":
                return await document_routes.query_embeddings_by_file_id(
                    QueryRequestBody(file_id="file-1", **body_args), req
                )
            return await document_routes.query_embeddings_by_file_ids(
                req, QueryMultipleBody(file_ids=["file-1", "file-2"], **body_args)
            )

        try:
            yield SimpleNamespace(
                query=query,
                embedding=embedding,
                search=search,
                hits=hits,
                request=req,
            )
        finally:
            # Drain cancelled read-only work before clearing its shared cache.
            pool.shutdown(wait=True)
            document_routes.get_cached_query_embedding.cache_clear()


def _block_stage(context, stage, loop, started, release, finished):
    operation = getattr(context, stage)
    result = operation.return_value

    def blocking(*args, **kwargs):
        assert threading.current_thread().name.startswith("query-test")
        loop.call_soon_threadsafe(started.set)
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("event loop did not release the worker")
            return result
        finally:
            loop.call_soon_threadsafe(finished.set)

    operation.side_effect = blocking


@pytest.mark.parametrize("stage", ["embedding", "search"])
async def test_blocking_stages_leave_event_loop_responsive(query_context, stage):
    started, finished = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    _block_stage(
        query_context, stage, asyncio.get_running_loop(), started, release, finished
    )
    task = asyncio.create_task(query_context.query())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        # This coroutine can resume while the provider or database is still blocked.
        assert not task.done()
        assert not finished.is_set()
    finally:
        release.set()
        result = await asyncio.wait_for(task, timeout=2)
    assert result == query_context.hits


async def test_concurrent_queries_share_the_configured_worker_limit(query_context):
    loop = asyncio.get_running_loop()
    saturated = asyncio.Event()
    release = threading.Event()
    lock = threading.Lock()
    started = 0

    def embed(query):
        nonlocal started
        with lock:
            started += 1
            if started == 2:
                loop.call_soon_threadsafe(saturated.set)
        if not release.wait(timeout=5):
            raise TimeoutError("event loop did not release the workers")
        return [0.1, 0.2, 0.3]

    query_context.embedding.side_effect = embed
    tasks = [asyncio.create_task(query_context.query(str(i))) for i in range(5)]
    try:
        await asyncio.wait_for(saturated.wait(), timeout=2)
        # Two workers are occupied. The remaining requests must wait in this
        # same pool, not start in a separate or unbounded set of threads.
        with lock:
            assert started == 2
        assert all(not task.done() for task in tasks)
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
    assert results == [query_context.hits] * 5
    assert started == 5


async def test_cache_reuses_vectors_but_not_scoped_results(query_context, query_route):
    assert await query_context.query() == query_context.hits
    query_context.request.state.user = {"id": "owner-2"}
    assert await query_context.query(entity_id="agent-1") == query_context.hits
    query_context.embedding.assert_called_once_with("question")
    assert query_context.search.call_count == 2
    expected_file_clause = (
        {"file_id": {"$eq": "file-1"}}
        if query_route == "single"
        else {"file_id": {"$in": ["file-1", "file-2"]}}
    )
    for call, owners in zip(
        query_context.search.call_args_list,
        [["owner-1"], ["agent-1", "owner-2"]],
    ):
        # The real async adapter forwards positional arguments; the sync path
        # forwards keyword arguments. Both must carry the same pre-ranking scope.
        embedding = call.args[0]
        k = call.args[1] if len(call.args) > 1 else call.kwargs["k"]
        predicate = call.args[2] if len(call.args) > 2 else call.kwargs["filter"]
        assert embedding == [0.1, 0.2, 0.3]
        assert k == 7
        assert predicate == {
            "$and": [expected_file_clause, {"user_id": {"$in": owners}}]
        }


@pytest.mark.parametrize("stage", ["embedding", "search"])
@pytest.mark.parametrize(
    "error,status",
    [
        (ValueError("unavailable"), 500),
        (HTTPException(429, "retry"), 429),
        (StopIteration("exhausted"), 500),
    ],
)
async def test_worker_errors_propagate_and_allow_retry(
    query_context, stage, error, status
):
    operation = getattr(query_context, stage)
    operation.side_effect = error
    with pytest.raises(HTTPException) as caught:
        await asyncio.wait_for(query_context.query(), timeout=2)
    assert caught.value.status_code == status
    if stage == "embedding":
        query_context.search.assert_not_called()
    operation.side_effect = None
    assert await query_context.query() == query_context.hits
    assert operation.call_count == 2


@pytest.mark.parametrize("stage", ["embedding", "search"])
async def test_cancellation_propagates_while_read_only_worker_finishes(
    query_context, stage
):
    started, finished = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    _block_stage(
        query_context, stage, asyncio.get_running_loop(), started, release, finished
    )
    task = asyncio.create_task(query_context.query())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=2)
        await asyncio.gather(task, return_exceptions=True)
    if stage == "embedding":
        query_context.search.assert_not_called()


@pytest.mark.parametrize("db_type", [VectorDBType.PGVECTOR, VectorDBType.ATLAS_MONGO])
async def test_distance_threshold_semantics_unchanged(
    query_context, monkeypatch, db_type
):
    monkeypatch.setattr(document_routes, "VECTOR_DB_TYPE", db_type)
    monkeypatch.setattr(document_routes, "RAG_DISTANCE_THRESHOLD", 0.3)
    expected = (
        query_context.hits[:1]
        if db_type == VectorDBType.PGVECTOR
        else query_context.hits
    )
    assert await query_context.query() == expected


async def test_empty_result_contract_unchanged(query_context, query_route):
    query_context.search.return_value = []
    if query_route == "single":
        assert await query_context.query() == []
    else:
        with pytest.raises(HTTPException) as caught:
            await query_context.query()
        assert caught.value.status_code == 404
        assert caught.value.detail == "No documents found for the given query"
