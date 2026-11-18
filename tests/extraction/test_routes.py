import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import httpx
import jwt
import pytest
from fastapi import FastAPI
from langchain_core.documents import Document

from app.config import LogMiddleware
from app.middleware import security_middleware
from app.routes import document_routes as routes
from app.services.extraction_client import ExtractionClient, ExtractionSettings
from app.services.vector_store.async_pg_vector import AsyncPgVector

ROUTES = ["/text", "/embed", "/embed-upload", "/local/embed"]
MIMES = {
    "txt": "text/plain",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class RecordingStore(AsyncPgVector):
    def __init__(self):
        self._bind = None
        self.calls = []
        self.persisted = []
        self.rollbacks = []
        self.fail_on = None

    async def aadd_documents(self, documents, ids=None, executor=None):
        self.calls.append(copy.deepcopy(documents))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("insertion failed")
        self.persisted.extend(copy.deepcopy(documents))
        return [f"row-{len(self.calls)}-{index}" for index in range(len(documents))]

    async def delete_by_metadata(self, metadata, executor=None):
        self.rollbacks.append(metadata)
        self.persisted = [
            doc
            for doc in self.persisted
            if not all(
                doc.metadata.get(key) == value for key, value in metadata.items()
            )
        ]

    async def get_filtered_ids(self, ids, owners=None, executor=None):
        return [
            file_id
            for file_id in ids
            if any(
                doc.metadata["file_id"] == file_id and doc.metadata["user_id"] in owners
                for doc in self.persisted
            )
        ]

    async def get_documents_by_ids(self, ids, owners=None, executor=None):
        return [
            doc
            for doc in reversed(self.persisted)
            if doc.metadata["file_id"] in ids and doc.metadata["user_id"] in owners
        ]


@pytest.fixture
async def api(processor, monkeypatch, tmp_path):
    app = FastAPI()
    app.add_middleware(LogMiddleware)
    app.middleware("http")(security_middleware)
    app.include_router(routes.router)
    store = RecordingStore()
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(routes, "RAG_UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(routes, "extraction_client", processor.client())
    monkeypatch.setattr(routes, "PDF_EXTRACT_IMAGES", False)
    monkeypatch.setattr(routes, "vector_store", store)
    monkeypatch.setattr(routes, "PARALLEL_EXECUTION", 1)
    monkeypatch.setenv("JWT_SECRET", "test-extraction-secret")
    # The migrated path must never construct either legacy parser.
    local_loader = routes.get_loader

    def no_migrated_local_loader(filename, content_type, *args, **kwargs):
        assert (
            routes.extraction_format(filename, content_type) is None
        ), "local migrated parser"
        return local_loader(filename, content_type, *args, **kwargs)

    monkeypatch.setattr(routes, "get_loader", no_migrated_local_loader)
    monkeypatch.setattr(
        "app.utils.document_loader.Docx2txtLoader",
        Mock(side_effect=AssertionError("local DOCX parser")),
    )
    token = jwt.encode(
        {
            "id": "caller",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        "test-extraction-secret",
        algorithm="HS256",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        app.state.thread_pool = executor
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://rag",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            yield client, store, uploads


async def post(
    client,
    uploads,
    route,
    format="pdf",
    content=b"opaque bytes",
    mime=None,
    filename=None,
    entity=None,
):
    filename = filename or f"report.{format}"
    mime = mime or MIMES.get(format, "text/plain")
    if route == "/local/embed":
        local = uploads / filename
        local.write_bytes(content)
        return await client.post(
            route,
            params={"entity_id": entity} if entity else None,
            json={
                "filepath": filename,
                "filename": filename,
                "file_content_type": mime,
                "file_id": "caller-file",
            },
        )
    data = {"file_id": "caller-file"}
    if entity:
        data["entity_id"] = entity
    return await client.post(
        route,
        data=data,
        files={
            "uploaded_file" if route == "/embed-upload" else "file": (
                filename,
                content,
                mime,
            ),
        },
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_migrated_routes_parse_once_and_keep_caller_metadata(
    api, processor, route, format
):
    client, store, uploads = api
    response = await post(client, uploads, route, format)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["file_id"] == "caller-file"
    assert result["filename"] == f"report.{format}"
    assert result["known_type"] is True
    assert [r[0] for r in processor.requests] == ["GET", "POST"]
    assert all(
        request[2]["Authorization"] == f"Bearer {processor.token}"
        and request[2]["Authorization"] != client.headers["Authorization"]
        for request in processor.requests
    )
    if route == "/text":
        assert result["text"] == (
            "First page café\nSecond page"
            if format == "pdf"
            else "Body paragraph\nLeft\tRight"
        )
        assert not store.calls
    else:
        assert result["status"] is True
        docs = store.persisted
        assert docs
        assert all(
            doc.metadata["user_id"] == "caller"
            and doc.metadata["file_id"] == "caller-file"
            for doc in docs
        )
        assert all(str(uploads) in doc.metadata["source"] for doc in docs)
        assert all("\x00" not in doc.page_content for doc in docs)
        assert [doc.metadata["_rag_chunk_index"] for doc in docs] == list(
            range(len(docs))
        )
        assert len({doc.metadata["_rag_ingestion_attempt_id"] for doc in docs}) == 1
        if format == "pdf":
            assert [doc.metadata["page"] for doc in docs] == [0, 1]
            context = await client.get("/documents/caller-file/context")
            assert context.status_code == 200
            assert context.text.index("First page") < context.text.index("Second page")
            assert "# PAGE 0" not in context.text
            assert "# PAGE 1" in context.text  # Existing zero-based context convention.
        else:
            assert all("page" not in doc.metadata for doc in docs)
    remaining = list(uploads.rglob("*.*"))
    assert len(remaining) == (1 if route == "/local/embed" else 0)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "failure,status",
    [
        ("unsupported", 422),
        ("outage", 503),
        ("incompatible", 502),
        ("identity", 502),
    ],
)
async def test_selected_format_failure_never_falls_back_or_persists(
    api, processor, route, failure, status
):
    client, store, uploads = api
    if failure == "unsupported":
        processor.error = (422, "UNSUPPORTED_CONTENT")
    elif failure == "outage":
        routes.extraction_client = ExtractionClient(
            ExtractionSettings(token_file="/run/extraction/missing-token")
        )
    elif failure == "incompatible":
        processor.capabilities["version"] = 2
    else:
        processor.mutate = lambda value: {
            **value,
            "user_id": "forged",
            "file_id": "forged",
            "_rag_chunk_index": 99,
        }
    response = await post(client, uploads, route)
    assert response.status_code == status, response.text
    assert not store.calls and not store.persisted and not store.rollbacks
    assert len(list(uploads.rglob("*.pdf"))) == (1 if route == "/local/embed" else 0)


@pytest.mark.parametrize("route", ROUTES)
async def test_unauthorized_request_cannot_reach_processor(api, processor, route):
    client, store, uploads = api
    client.headers.pop("Authorization")
    response = await post(client, uploads, route)
    assert response.status_code == 401
    assert not processor.requests and not store.calls


@pytest.mark.parametrize("route", ROUTES)
async def test_cancellation_cleans_upload_and_never_stores(api, processor, route):
    client, store, uploads = api
    processor.hold = True
    task = asyncio.create_task(post(client, uploads, route))
    await asyncio.wait_for(processor.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(processor.closed.wait(), 1)
    assert not store.calls
    assert len(list(uploads.rglob("*.pdf"))) == (1 if route == "/local/embed" else 0)


@pytest.mark.parametrize("route", ["/embed", "/embed-upload", "/local/embed"])
async def test_existing_attempt_rollback_and_no_success_on_storage_error(
    api, processor, monkeypatch, route
):
    client, store, uploads = api
    monkeypatch.setattr(routes, "EMBEDDING_BATCH_SIZE", 1)
    store.fail_on = 2
    old = Document(
        page_content="earlier attempt",
        metadata={
            "file_id": "caller-file",
            "user_id": "caller",
            "_rag_ingestion_attempt_id": "old",
        },
    )
    store.persisted.append(old)
    response = await post(client, uploads, route)
    assert response.status_code == 500, response.text
    assert store.persisted == [old]
    assert len(store.rollbacks) == 1
    assert store.rollbacks[0]["file_id"] == "caller-file"
    assert store.rollbacks[0]["_rag_ingestion_attempt_id"] != "old"
    assert len([r for r in processor.requests if r[0] == "POST"]) == 1


@pytest.mark.parametrize(
    "filename,mime",
    [
        ("notes.md", "text/markdown"),
        ("plain.txt", "text/plain"),
        ("code.py", "text/x-python"),
    ],
)
async def test_raw_text_parity_keeps_original_text_and_local_path(
    api, processor, filename, mime
):
    client, store, uploads = api
    content = "# Heading\n\n**bold** and café\n- item\n".encode()
    response = await post(client, uploads, "/text", "txt", content, mime, filename)
    assert response.status_code == 200
    assert response.json()["text"] == content.decode().rstrip("\n")
    assert response.json()["known_type"] is True
    assert not processor.requests and not store.calls


async def test_mime_selection_returns_canonical_pdf_and_cleans_text(api, processor):
    client, store, uploads = api
    response = await post(
        client, uploads, "/text", mime="application/pdf", filename="opaque.bin"
    )
    assert response.status_code == 200
    assert "\x00" not in response.json()["text"]
    assert processor.requests[-1][2]["X-Extraction-Format"] == "pdf"


async def test_selected_docx_cannot_be_hijacked_by_markdown_mime(api, processor):
    client, store, uploads = api
    response = await post(client, uploads, "/text", "docx", mime="text/markdown")
    assert response.status_code == 200
    assert processor.requests[-1][2]["X-Extraction-Format"] == "docx"


async def test_requested_pdf_image_extraction_is_not_silently_downgraded(
    api, processor, monkeypatch
):
    client, store, uploads = api
    monkeypatch.setattr(routes, "PDF_EXTRACT_IMAGES", True)
    response = await post(client, uploads, "/text")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "UNSUPPORTED_CONTENT"
    assert not processor.requests and not store.calls


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_no_extraction_configuration_still_requires_service(
    api, processor, monkeypatch, route, format
):
    client, store, uploads = api
    for key in (
        "RAG_EXTRACTION_ENABLED",
        "RAG_EXTRACTION_FORMATS",
        "RAG_EXTRACTION_SOCKET",
        "RAG_EXTRACTION_URL",
        "RAG_EXTRACTION_TOKEN_FILE",
        "RAG_EXTRACTION_DEADLINE_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_EXTRACTION_TOKEN_FILE", processor.token_file)
    defaults = ExtractionSettings.from_env()
    routes.extraction_client = ExtractionClient(defaults)
    connect = asyncio.open_connection

    async def disposable_endpoint(host, port, **kwargs):
        assert (host, port) == ("extraction", 8000)
        return await connect(
            "127.0.0.1", int(processor.url.rsplit(":", 1)[1]), **kwargs
        )

    monkeypatch.setattr(asyncio, "open_connection", disposable_endpoint)
    response = await post(client, uploads, route, format)
    assert response.status_code == 200, response.text
    assert [r[0] for r in processor.requests] == ["GET", "POST"]
    assert response.json()["known_type"] is True


class ASGIConnection:
    """Deliver the complete upload, then a server disconnect without cancelling app."""

    def __init__(self, app, request, body):
        self.incoming = asyncio.Queue()
        self.received = []
        self.sent = []
        chunks = [body[:17], body[17:53], body[53:]]
        for index, chunk in enumerate(chunks):
            self.incoming.put_nowait(
                {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": index < len(chunks) - 1,
                }
            )
        self.scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": request.method,
            "scheme": "http",
            "path": request.url.path,
            "raw_path": request.url.raw_path,
            "query_string": request.url.query,
            "root_path": "",
            "headers": [(key.lower(), value) for key, value in request.headers.raw],
            "client": ("127.0.0.1", 12345),
            "server": ("rag", 80),
        }
        self.task = asyncio.create_task(app(self.scope, self.receive, self.send))

    async def receive(self):
        message = await self.incoming.get()
        self.received.append(message["type"])
        return message

    async def send(self, message):
        self.sent.append(message)

    def disconnect(self):
        self.incoming.put_nowait({"type": "http.disconnect"})

    async def close(self):
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def disconnect_request(client, uploads, route, format):
    filename = f"report.{format}"
    if route == "/local/embed":
        (uploads / filename).write_bytes(b"opaque document bytes")
        return client.build_request(
            "POST",
            route,
            json={
                "filepath": filename,
                "filename": filename,
                "file_content_type": MIMES[format],
                "file_id": "caller-file",
            },
        )
    return client.build_request(
        "POST",
        route,
        data={"file_id": "caller-file"},
        files={
            "uploaded_file" if route == "/embed-upload" else "file": (
                filename,
                b"opaque document bytes",
                MIMES[format],
            ),
        },
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_asgi_disconnect_stops_extraction_and_cleans_upload(
    api, processor, route, format
):
    client, store, uploads = api
    processor.hold = True
    request = disconnect_request(client, uploads, route, format)
    connection = ASGIConnection(client._transport.app, request, await request.aread())
    try:
        await asyncio.wait_for(processor.entered.wait(), 1)
        assert connection.received == ["http.request"] * 3
        connection.disconnect()
        await asyncio.wait_for(processor.closed.wait(), 0.5)
        await asyncio.wait_for(asyncio.shield(connection.task), 1)
        assert "http.disconnect" in connection.received
        assert connection.task.cancelling() == 0
        assert not store.calls and not store.persisted and not store.rollbacks
        assert not any(
            message["type"] == "http.response.start" and 200 <= message["status"] < 300
            for message in connection.sent
        )
        assert len(list(uploads.rglob(f"*.{format}"))) == (
            1 if route == "/local/embed" else 0
        )
        assert routes.extraction_client._admitted == 0
    finally:
        await connection.close()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_asgi_disconnect_wins_over_processor_success(
    api, processor, route, format
):
    client, store, uploads = api
    request = disconnect_request(client, uploads, route, format)
    body = await request.aread()
    processor.before_response = lambda: connection.disconnect()
    connection = ASGIConnection(client._transport.app, request, body)
    try:
        await asyncio.wait_for(asyncio.shield(connection.task), 1)
        assert connection.received == ["http.request"] * 3 + ["http.disconnect"]
        assert connection.task.cancelling() == 0
        assert not store.calls and not store.persisted and not store.rollbacks
        assert not any(
            message["type"] == "http.response.start" and 200 <= message["status"] < 300
            for message in connection.sent
        )
        assert len(list(uploads.rglob(f"*.{format}"))) == (
            1 if route == "/local/embed" else 0
        )
        assert routes.extraction_client._admitted == 0
    finally:
        await connection.close()


@pytest.mark.parametrize("route", ["/embed", "/embed-upload", "/local/embed"])
@pytest.mark.parametrize("phase", ["preparation", "insertion"])
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_disconnect_after_extraction_drains_and_preserves_old_rows(
    api, processor, monkeypatch, route, phase, format
):
    import threading

    client, store, uploads = api
    monkeypatch.setattr(routes, "EMBEDDING_BATCH_SIZE", 1)
    old = Document(
        page_content="previous attempt",
        metadata={
            "file_id": "caller-file",
            "user_id": "caller",
            "_rag_ingestion_attempt_id": "old",
        },
    )
    store.persisted.append(old)
    events = []
    preparation_entered = threading.Event()
    preparation_release = threading.Event()
    insert_entered = asyncio.Event()
    insert_release = asyncio.Event()
    prepare = routes._prepare_documents_sync
    insert = store.aadd_documents
    rollback = store.delete_by_metadata

    def delayed_prepare(*args):
        events.append("preparation:start")
        preparation_entered.set()
        assert preparation_release.wait(3), "test preparation gate timed out"
        result = prepare(*args)
        events.append("preparation:done")
        return result

    async def delayed_insert(*args, **kwargs):
        events.append("insert:start")
        insert_entered.set()
        await insert_release.wait()
        result = await insert(*args, **kwargs)
        events.append("insert:commit")
        return result

    async def track_rollback(*args, **kwargs):
        events.append("rollback")
        return await rollback(*args, **kwargs)

    if phase == "preparation":
        monkeypatch.setattr(routes, "_prepare_documents_sync", delayed_prepare)
    else:
        monkeypatch.setattr(store, "aadd_documents", delayed_insert)
    monkeypatch.setattr(store, "delete_by_metadata", track_rollback)
    request = disconnect_request(client, uploads, route, format)
    connection = ASGIConnection(client._transport.app, request, await request.aread())
    try:
        if phase == "preparation":
            assert await asyncio.to_thread(preparation_entered.wait, 1)
        else:
            await asyncio.wait_for(insert_entered.wait(), 1)
        assert [r[0] for r in processor.requests] == ["GET", "POST"]
        events.append("disconnect:queued")
        connection.disconnect()
        await asyncio.sleep(0.05)
        assert not connection.task.done()
        assert not store.rollbacks
        assert list(uploads.rglob(f"*.{format}"))
        preparation_release.set()
        insert_release.set()
        await asyncio.wait_for(asyncio.shield(connection.task), 1)
        statuses = [
            message["status"]
            for message in connection.sent
            if message["type"] == "http.response.start"
        ]
        assert store.persisted == [old]
        assert statuses == [499]
        assert connection.task.cancelling() == 0
        if phase == "preparation":
            assert not store.calls and not store.rollbacks
            assert events.index("preparation:done") > events.index("disconnect:queued")
        else:
            assert events == [
                "insert:start",
                "disconnect:queued",
                "insert:commit",
                "rollback",
            ]
            assert store.rollbacks[0]["_rag_ingestion_attempt_id"] != "old"
        assert len(list(uploads.rglob(f"*.{format}"))) == (
            1 if route == "/local/embed" else 0
        )
    finally:
        preparation_release.set()
        insert_release.set()
        await connection.close()


class SyncRecordingStore:
    def __init__(self, entered, release, events, old):
        self.entered = entered
        self.release = release
        self.events = events
        self.calls = []
        self.persisted = [old]
        self.rollbacks = []

    def add_documents(self, documents, ids=None):
        self.calls.append(copy.deepcopy(documents))
        self.events.append("insert:start")
        self.entered.set()
        assert self.release.wait(3), "test sync insertion gate timed out"
        self.persisted.extend(copy.deepcopy(documents))
        self.events.append("insert:commit")
        return [
            f"sync-row-{len(self.calls)}-{index}" for index in range(len(documents))
        ]

    def delete_scoped(self, *args, **kwargs):
        self.rollbacks.append((args, kwargs))
        raise AssertionError("disconnect must not delete previous attempts by file_id")


@pytest.mark.parametrize("route", ["/embed", "/embed-upload", "/local/embed"])
@pytest.mark.parametrize("mode", ["async_single", "sync_single", "sync_batch"])
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_disconnect_after_uncancellable_storage_starts_drains_without_undo(
    api, processor, monkeypatch, route, mode, format
):
    import threading

    client, store, uploads = api
    monkeypatch.setattr(
        routes, "EMBEDDING_BATCH_SIZE", 1 if mode == "sync_batch" else 0
    )
    old = Document(
        page_content="previous attempt",
        metadata={
            "file_id": "caller-file",
            "user_id": "caller",
            "_rag_ingestion_attempt_id": "old",
        },
    )
    events = []
    sync_entered = threading.Event()
    sync_release = threading.Event()
    async_entered = asyncio.Event()
    async_release = asyncio.Event()
    if mode.startswith("sync"):
        store = SyncRecordingStore(sync_entered, sync_release, events, old)
        monkeypatch.setattr(routes, "vector_store", store)
    else:
        store.persisted.append(old)
        insert = store.aadd_documents

        async def delayed_insert(*args, **kwargs):
            events.append("insert:start")
            async_entered.set()
            await async_release.wait()
            result = await insert(*args, **kwargs)
            events.append("insert:commit")
            return result

        monkeypatch.setattr(store, "aadd_documents", delayed_insert)
    request = disconnect_request(client, uploads, route, format)
    connection = ASGIConnection(client._transport.app, request, await request.aread())
    try:
        if mode.startswith("sync"):
            assert await asyncio.to_thread(sync_entered.wait, 1)
        else:
            await asyncio.wait_for(async_entered.wait(), 1)
        events.append("disconnect:queued")
        connection.disconnect()
        await asyncio.sleep(0.05)
        assert "http.disconnect" in connection.received
        assert not connection.task.done()
        assert store.persisted == [old] and not store.rollbacks
        assert list(uploads.rglob(f"*.{format}"))
        sync_release.set()
        async_release.set()
        await asyncio.wait_for(asyncio.shield(connection.task), 1)
        assert old in store.persisted
        assert len(store.persisted) == (3 if format == "pdf" else 2)
        assert not store.rollbacks
        assert events.index("insert:commit") > events.index("disconnect:queued")
        expected_calls = 2 if mode == "sync_batch" and format == "pdf" else 1
        assert len(store.calls) == expected_calls
        assert [
            m["status"] for m in connection.sent if m["type"] == "http.response.start"
        ] == [499]
        assert connection.task.cancelling() == 0
        assert len(list(uploads.rglob(f"*.{format}"))) == (
            1 if route == "/local/embed" else 0
        )
    finally:
        sync_release.set()
        async_release.set()
        await connection.close()


@pytest.mark.parametrize("route", ["/embed", "/embed-upload", "/local/embed"])
async def test_unmigrated_text_keeps_post_extraction_disconnect_behavior(
    api, monkeypatch, route
):
    import threading

    client, store, uploads = api
    loader = Mock()
    loader.lazy_load.return_value = [Document(page_content="legacy text")]
    monkeypatch.setattr(routes, "get_loader", Mock(return_value=(loader, True, "txt")))
    entered = threading.Event()
    release = threading.Event()
    prepare = routes._prepare_documents_sync

    def delayed_prepare(*args):
        entered.set()
        assert release.wait(3), "test legacy preparation gate timed out"
        return prepare(*args)

    monkeypatch.setattr(routes, "_prepare_documents_sync", delayed_prepare)
    request = disconnect_request(client, uploads, route, "txt")
    connection = ASGIConnection(client._transport.app, request, await request.aread())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        connection.disconnect()
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.wait_for(asyncio.shield(connection.task), 1)
        assert "http.disconnect" not in connection.received
        assert len(store.persisted) == 1
        assert not store.rollbacks
        assert [
            m["status"] for m in connection.sent if m["type"] == "http.response.start"
        ] == [200]
    finally:
        release.set()
        await connection.close()


@pytest.mark.parametrize("disconnect", [False, True])
async def test_selected_sync_partial_failure_retains_old_and_committed_rows(
    api, monkeypatch, disconnect
):
    import threading

    client, _, uploads = api
    monkeypatch.setattr(routes, "EMBEDDING_BATCH_SIZE", 1)
    entered = threading.Event()
    release = threading.Event()
    events = []
    old = Document(
        page_content="old", metadata={"file_id": "caller-file", "user_id": "caller"}
    )
    store = SyncRecordingStore(entered, release, events, old)
    insert = store.add_documents

    def fail_second_batch(documents, ids=None):
        if store.calls:
            raise RuntimeError("second sync batch failed")
        return insert(documents, ids)

    monkeypatch.setattr(store, "add_documents", fail_second_batch)
    monkeypatch.setattr(routes, "vector_store", store)
    request = disconnect_request(client, uploads, "/embed", "pdf")
    connection = ASGIConnection(client._transport.app, request, await request.aread())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        if disconnect:
            connection.disconnect()
            await asyncio.sleep(0.05)
            assert "http.disconnect" in connection.received
        release.set()
        await asyncio.wait_for(asyncio.shield(connection.task), 1)
        assert len(store.persisted) == 2 and old in store.persisted
        assert not store.rollbacks
        assert [
            m["status"] for m in connection.sent if m["type"] == "http.response.start"
        ] == [499 if disconnect else 500]
    finally:
        release.set()
        await connection.close()


async def test_disconnect_during_upload_cleanup_waits_for_removal(api, monkeypatch):
    client, _, uploads = api
    entered = asyncio.Event()
    release = asyncio.Event()
    remove = routes.aiofiles.os.remove

    async def delayed_remove(*args, **kwargs):
        entered.set()
        await release.wait()
        await remove(*args, **kwargs)

    monkeypatch.setattr(routes.aiofiles.os, "remove", delayed_remove)
    request = disconnect_request(client, uploads, "/text", "pdf")
    connection = ASGIConnection(client._transport.app, request, await request.aread())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        connection.disconnect()
        await asyncio.sleep(0.05)
        assert not connection.task.done()
        release.set()
        await asyncio.wait_for(asyncio.shield(connection.task), 1)
        assert not list(uploads.rglob("*.pdf"))
        assert [
            m["status"] for m in connection.sent if m["type"] == "http.response.start"
        ] == [499]
    finally:
        release.set()
        await connection.close()


@pytest.mark.parametrize("batch_size", [0, 1])
async def test_disconnect_before_queued_sync_mutation_prevents_write(
    api, monkeypatch, batch_size
):
    import threading

    client, _, uploads = api
    app = client._transport.app
    original_executor = app.state.thread_pool
    monkeypatch.setattr(routes, "EMBEDDING_BATCH_SIZE", batch_size)
    blocker_entered = threading.Event()
    blocker_release = threading.Event()
    submitted = asyncio.Event()
    old = Document(
        page_content="old", metadata={"file_id": "caller-file", "user_id": "caller"}
    )
    store = SyncRecordingStore(threading.Event(), threading.Event(), [], old)
    monkeypatch.setattr(routes, "vector_store", store)
    prepare = routes._prepare_documents_sync
    run_sync = routes._run_sync_request_store

    def blocker():
        blocker_entered.set()
        assert blocker_release.wait(3), "test executor gate timed out"

    def prepare_then_block_executor(*args):
        docs = prepare(*args)
        app.state.thread_pool.submit(blocker)
        return docs

    async def observe_submission(*args):
        submitted.set()
        return await run_sync(*args)

    monkeypatch.setattr(routes, "_prepare_documents_sync", prepare_then_block_executor)
    monkeypatch.setattr(routes, "_run_sync_request_store", observe_submission)
    with ThreadPoolExecutor(max_workers=1) as executor:
        app.state.thread_pool = executor
        request = disconnect_request(client, uploads, "/embed", "pdf")
        connection = ASGIConnection(app, request, await request.aread())
        try:
            await asyncio.wait_for(submitted.wait(), 1)
            assert await asyncio.to_thread(blocker_entered.wait, 1)
            connection.disconnect()
            await asyncio.sleep(0.05)
            assert "http.disconnect" in connection.received
            assert not connection.task.done()
            assert not store.calls
            blocker_release.set()
            await asyncio.wait_for(asyncio.shield(connection.task), 1)
            assert store.persisted == [old]
            assert not store.calls and not store.rollbacks
            assert [
                m["status"]
                for m in connection.sent
                if m["type"] == "http.response.start"
            ] == [499]
        finally:
            blocker_release.set()
            await connection.close()
            app.state.thread_pool = original_executor


@pytest.fixture
def freeze_disconnect_wrapper(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    original_wait = asyncio.wait
    state = {}

    async def frozen_wait(tasks, **kwargs):
        result = await original_wait(tasks, **kwargs)
        watcher = next(
            (
                task
                for task in tasks
                if "_with_extraction_request" in task.get_coro().__qualname__
                and task.get_coro().__name__ == "wait_for_disconnect"
            ),
            None,
        )
        if watcher in result[0]:
            state["operation"] = next(task for task in tasks if task is not watcher)
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(asyncio, "wait", frozen_wait)
    return entered, release, state


@pytest.mark.parametrize("route", ["/embed", "/embed-upload", "/local/embed"])
@pytest.mark.parametrize("format", ["pdf", "docx"])
@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_completed_batch_disconnect_cannot_cancel_queued_attempt_rollback(
    api, monkeypatch, freeze_disconnect_wrapper, route, format, repeat_cancel
):
    import threading

    client, store, uploads = api
    app = client._transport.app
    original_executor = app.state.thread_pool
    monkeypatch.setattr(routes, "EMBEDDING_BATCH_SIZE", 500)
    wrapper_entered, wrapper_release, state = freeze_disconnect_wrapper
    insert_entered = asyncio.Event()
    insert_release = asyncio.Event()
    rollback_queued = asyncio.Event()
    worker_entered = threading.Event()
    worker_release = threading.Event()
    events = []
    queued_deletes = []
    old = Document(
        page_content="old attempt",
        metadata={
            "file_id": "caller-file",
            "user_id": "caller",
            "_rag_ingestion_attempt_id": "old",
        },
    )
    store.persisted.append(old)
    insert = store.aadd_documents

    def occupy_worker():
        worker_entered.set()
        assert worker_release.wait(3), "test rollback executor gate timed out"

    async def final_insert(*args, executor=None, **kwargs):
        executor.submit(occupy_worker)
        insert_entered.set()
        await insert_release.wait()
        result = await insert(*args, executor=executor, **kwargs)
        events.append("insert:commit")
        return result

    def delete_current_attempt(metadata):
        events.append("rollback:execute")
        store.rollbacks.append(metadata)
        store.persisted = [
            doc
            for doc in store.persisted
            if not all(
                doc.metadata.get(key) == value for key, value in metadata.items()
            )
        ]

    async def queued_rollback(metadata, executor=None):
        future = asyncio.get_running_loop().run_in_executor(
            executor, delete_current_attempt, metadata
        )
        queued_deletes.append(future)
        events.append("rollback:queued")
        rollback_queued.set()
        await future

    monkeypatch.setattr(store, "aadd_documents", final_insert)
    monkeypatch.setattr(store, "delete_by_metadata", queued_rollback)
    with ThreadPoolExecutor(max_workers=1) as executor:
        app.state.thread_pool = executor
        request = disconnect_request(client, uploads, route, format)
        connection = ASGIConnection(app, request, await request.aread())
        try:
            await asyncio.wait_for(insert_entered.wait(), 1)
            assert await asyncio.to_thread(worker_entered.wait, 1)
            connection.disconnect()
            await asyncio.wait_for(wrapper_entered.wait(), 1)
            insert_release.set()
            await asyncio.wait_for(rollback_queued.wait(), 1)
            assert len(store.persisted) > 1 and not store.rollbacks
            wrapper_release.set()
            await asyncio.sleep(0.05)
            assert state["operation"].cancelling() == 0
            if repeat_cancel:
                for _ in range(2):
                    state["operation"].cancel()
                    await asyncio.sleep(0)
                await asyncio.sleep(0.05)
            assert not connection.task.done(), events
            assert not queued_deletes[0].cancelled(), events
            assert list(uploads.rglob(f"*.{format}"))
            worker_release.set()
            await asyncio.wait_for(asyncio.shield(connection.task), 1)
            assert events == ["insert:commit", "rollback:queued", "rollback:execute"]
            assert store.persisted == [old]
            assert len(store.rollbacks) == 1
            assert store.rollbacks[0]["_rag_ingestion_attempt_id"] != "old"
            assert connection.task.cancelling() == 0
            assert [
                m["status"]
                for m in connection.sent
                if m["type"] == "http.response.start"
            ] == [499]
            assert len(list(uploads.rglob(f"*.{format}"))) == (
                1 if route == "/local/embed" else 0
            )
        finally:
            wrapper_release.set()
            insert_release.set()
            worker_release.set()
            await connection.close()
            app.state.thread_pool = original_executor


@pytest.mark.parametrize("route", ["/embed", "/embed-upload", "/local/embed"])
@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_disconnect_drain_owns_pending_insert_until_attempt_rollback(
    api, monkeypatch, freeze_disconnect_wrapper, route, repeat_cancel
):
    import threading

    client, store, uploads = api
    app = client._transport.app
    original_executor = app.state.thread_pool
    monkeypatch.setattr(routes, "EMBEDDING_BATCH_SIZE", 1)
    monkeypatch.setattr(routes, "PARALLEL_EXECUTION", 2)
    wrapper_entered, wrapper_release, state = freeze_disconnect_wrapper
    second_batch = asyncio.Event()
    insert_draining = asyncio.Event()
    worker_entered = threading.Event()
    worker_release = threading.Event()
    insert_tasks = []
    insert_futures = []
    events = []
    old = Document(
        page_content="old",
        metadata={
            "file_id": "caller-file",
            "user_id": "caller",
            "_rag_ingestion_attempt_id": "old",
        },
    )
    store.persisted.append(old)
    original_queue = asyncio.Queue
    original_gather = asyncio.gather

    class ControlledQueue(original_queue):
        async def get(self):
            item = await super().get()
            if isinstance(item, tuple) and len(item) == 4 and item[2] >= 2:
                await second_batch.wait()
            return item

    def observe_gather(*tasks, **kwargs):
        if insert_tasks and insert_tasks[0] in tasks:
            insert_draining.set()
        return original_gather(*tasks, **kwargs)

    def commit_after_release(documents):
        events.append("insert:start")
        worker_entered.set()
        assert worker_release.wait(3), "test insert executor gate timed out"
        store.calls.append(copy.deepcopy(documents))
        store.persisted.extend(copy.deepcopy(documents))
        events.append("insert:commit")
        return ["new-row"]

    async def blocked_insert(documents, ids=None, executor=None):
        insert_tasks.append(asyncio.current_task())
        future = asyncio.get_running_loop().run_in_executor(
            executor, commit_after_release, documents
        )
        insert_futures.append(future)
        return await future

    def remove_attempt(metadata):
        events.append("rollback")
        store.rollbacks.append(metadata)
        store.persisted = [
            doc
            for doc in store.persisted
            if not all(
                doc.metadata.get(key) == value for key, value in metadata.items()
            )
        ]

    async def rollback(metadata, executor=None):
        await asyncio.get_running_loop().run_in_executor(
            executor, remove_attempt, metadata
        )

    monkeypatch.setattr(asyncio, "Queue", ControlledQueue)
    monkeypatch.setattr(asyncio, "gather", observe_gather)
    monkeypatch.setattr(store, "aadd_documents", blocked_insert)
    monkeypatch.setattr(store, "delete_by_metadata", rollback)
    with ThreadPoolExecutor(max_workers=1) as executor:
        app.state.thread_pool = executor
        request = disconnect_request(client, uploads, route, "pdf")
        connection = ASGIConnection(app, request, await request.aread())
        try:
            assert await asyncio.to_thread(worker_entered.wait, 1)
            connection.disconnect()
            await asyncio.wait_for(wrapper_entered.wait(), 1)
            second_batch.set()
            await asyncio.wait_for(insert_draining.wait(), 1)
            assert state["operation"].cancelling() == 0
            wrapper_release.set()
            await asyncio.sleep(0.05)
            assert state["operation"].cancelling() == 0
            if repeat_cancel:
                # The real disconnect started cleanup; stress subsequent owner cancellation.
                state["operation"].cancel()
                await asyncio.sleep(0)
                state["operation"].cancel()
                await asyncio.sleep(0.05)
            assert not connection.task.done()
            assert not insert_futures[0].cancelled()
            assert not insert_tasks[0].cancelled()
            assert store.persisted == [old] and not store.rollbacks
            assert list(uploads.rglob("*.pdf"))
            worker_release.set()
            await asyncio.wait_for(asyncio.shield(connection.task), 1)
            assert events == ["insert:start", "insert:commit", "rollback"]
            assert len(insert_tasks) == 1 and len(store.calls) == 1
            assert store.persisted == [old]
            assert store.rollbacks[0]["_rag_ingestion_attempt_id"] != "old"
            assert [
                m["status"]
                for m in connection.sent
                if m["type"] == "http.response.start"
            ] == [499]
            assert len(list(uploads.rglob("*.pdf"))) == (
                1 if route == "/local/embed" else 0
            )
            assert connection.task.cancelling() == 0
        finally:
            second_batch.set()
            wrapper_release.set()
            worker_release.set()
            await connection.close()
            app.state.thread_pool = original_executor


@pytest.mark.parametrize("route", ["/embed", "/embed-upload", "/local/embed"])
async def test_repeated_cancellation_during_preparation_still_drains_without_writes(
    api, monkeypatch, freeze_disconnect_wrapper, route
):
    import threading

    client, store, uploads = api
    wrapper_entered, wrapper_release, state = freeze_disconnect_wrapper
    worker_entered = threading.Event()
    worker_release = threading.Event()
    worker_done = threading.Event()
    futures = []
    prepare = routes._prepare_documents_sync
    drain = routes._drain_request_worker

    def blocked_prepare(*args):
        worker_entered.set()
        assert worker_release.wait(3), "test preparation executor gate timed out"
        result = prepare(*args)
        worker_done.set()
        return result

    async def observe_drain(future):
        if routes._active_extraction_request.get() is not None and not futures:
            futures.append(future)
        return await drain(future)

    monkeypatch.setattr(routes, "_prepare_documents_sync", blocked_prepare)
    monkeypatch.setattr(routes, "_drain_request_worker", observe_drain)
    request = disconnect_request(client, uploads, route, "pdf")
    connection = ASGIConnection(client._transport.app, request, await request.aread())
    try:
        assert await asyncio.to_thread(worker_entered.wait, 1)
        connection.disconnect()
        await asyncio.wait_for(wrapper_entered.wait(), 1)
        wrapper_release.set()
        await asyncio.sleep(0.05)
        for _ in range(2):
            state["operation"].cancel()
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        assert not connection.task.done() and not worker_done.is_set()
        assert not futures[0].cancelled()
        assert not store.calls and not store.persisted and not store.rollbacks
        assert list(uploads.rglob("*.pdf"))
        worker_release.set()
        await asyncio.wait_for(asyncio.shield(connection.task), 1)
        assert worker_done.is_set()
        assert not store.calls and not store.persisted and not store.rollbacks
        assert [
            m["status"] for m in connection.sent if m["type"] == "http.response.start"
        ] == [499]
        assert len(list(uploads.rglob("*.pdf"))) == (
            1 if route == "/local/embed" else 0
        )
        assert connection.task.cancelling() == 0
    finally:
        wrapper_release.set()
        worker_release.set()
        await connection.close()


async def test_repeated_cancellation_cannot_abandon_upload_unlink(
    api, monkeypatch, freeze_disconnect_wrapper
):
    client, _, uploads = api
    wrapper_entered, wrapper_release, state = freeze_disconnect_wrapper
    unlink_entered = asyncio.Event()
    unlink_release = asyncio.Event()
    unlink_tasks = []
    remove = routes.aiofiles.os.remove

    async def delayed_remove(*args, **kwargs):
        unlink_tasks.append(asyncio.current_task())
        unlink_entered.set()
        await unlink_release.wait()
        await remove(*args, **kwargs)

    monkeypatch.setattr(routes.aiofiles.os, "remove", delayed_remove)
    request = disconnect_request(client, uploads, "/text", "pdf")
    connection = ASGIConnection(client._transport.app, request, await request.aread())
    try:
        await asyncio.wait_for(unlink_entered.wait(), 1)
        connection.disconnect()
        await asyncio.wait_for(wrapper_entered.wait(), 1)
        wrapper_release.set()
        await asyncio.sleep(0.05)
        for _ in range(2):
            state["operation"].cancel()
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        assert not connection.task.done() and not unlink_tasks[0].cancelled()
        assert list(uploads.rglob("*.pdf"))
        unlink_release.set()
        await asyncio.wait_for(asyncio.shield(connection.task), 1)
        assert not list(uploads.rglob("*.pdf"))
        assert [
            m["status"] for m in connection.sent if m["type"] == "http.response.start"
        ] == [499]
    finally:
        wrapper_release.set()
        unlink_release.set()
        await connection.close()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_default_service_unavailable_fails_closed_without_affecting_text(
    api, processor, monkeypatch, route, format
):
    client, store, uploads = api
    for key in (
        "RAG_EXTRACTION_ENABLED",
        "RAG_EXTRACTION_FORMATS",
        "RAG_EXTRACTION_SOCKET",
        "RAG_EXTRACTION_URL",
        "RAG_EXTRACTION_TOKEN_FILE",
        "RAG_EXTRACTION_DEADLINE_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_EXTRACTION_TOKEN_FILE", processor.token_file)
    routes.extraction_client = ExtractionClient(ExtractionSettings.from_env())

    async def unavailable(host, port, **kwargs):
        assert (host, port) == ("extraction", 8000)
        raise ConnectionRefusedError()

    monkeypatch.setattr(asyncio, "open_connection", unavailable)
    response = await post(client, uploads, route, format)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "UNAVAILABLE"
    assert not store.calls and not store.persisted and not processor.requests
    assert len(list(uploads.rglob(f"*.{format}"))) == (
        1 if route == "/local/embed" else 0
    )
    plaintext = await post(client, uploads, "/text", "txt", b"plain text", "text/plain")
    assert plaintext.status_code == 200 and plaintext.json()["text"] == "plain text"


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("failure", ["missing", "wrong", "extract-unauthorized"])
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_service_auth_failure_is_unavailable_not_user_unauthorized(
    api, processor, route, failure, format, caplog
):
    client, store, uploads = api
    if failure == "missing":
        Path(processor.token_file).unlink()
    elif failure == "wrong":
        Path(processor.token_file).write_text("cd" * 32)
    else:
        processor.error = (401, "UNAUTHORIZED")
    response = await post(client, uploads, route, format)
    assert response.status_code == 503
    assert response.json()["detail"] == {"code": "UNAVAILABLE"}
    assert not store.calls and not store.persisted and not store.rollbacks
    assert (
        len(processor.requests)
        == {"missing": 0, "wrong": 1, "extract-unauthorized": 2}[failure]
    )
    assert "Bearer" not in caplog.text and processor.token not in caplog.text
    assert "cd" * 32 not in caplog.text
    assert len(list(uploads.rglob(f"*.{format}"))) == (
        1 if route == "/local/embed" else 0
    )
    plaintext = await post(client, uploads, "/text", "txt", b"plain text", "text/plain")
    assert plaintext.status_code == 200
