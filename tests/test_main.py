import os
import jwt
import datetime
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

from main import app
from app.routes import document_routes

client = TestClient(app)


@pytest.fixture
def auth_headers():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def override_vector_store(monkeypatch):
    from app.config import vector_store
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from app.routes import document_routes

    # Clear the LRU cache and patch the cached function to return dummy embeddings
    document_routes.get_cached_query_embedding.cache_clear()

    def dummy_get_cached_query_embedding(query):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        document_routes, "get_cached_query_embedding", dummy_get_cached_query_embedding
    )

    # Initialize thread pool for tests since TestClient doesn't run lifespan
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="test-worker"
        )

    # Override get_all_ids as an async function - patch at CLASS level to bypass run_in_executor
    async def dummy_get_all_ids(self, owners=None, executor=None):
        return ["testid1", "testid2"]

    monkeypatch.setattr(AsyncPgVector, "get_all_ids", dummy_get_all_ids)

    # Override get_filtered_ids as an async function.
    async def dummy_get_filtered_ids(self, ids, owners=None, executor=None):
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", dummy_get_filtered_ids)

    # Override get_documents_by_ids as an async function.
    async def dummy_get_documents_by_ids(self, ids, owners=None, executor=None):
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    monkeypatch.setattr(
        AsyncPgVector, "get_documents_by_ids", dummy_get_documents_by_ids
    )

    # Override embedding_function with a dummy that doesn't call OpenAI
    class DummyEmbedding:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    vector_store.embedding_function = DummyEmbedding()

    # Override similarity search to return a tuple (Document, score).
    def dummy_similarity_search_with_score_by_vector(self, embedding, k, filter):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1"),
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    async def dummy_asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1") if filter else "testid1",
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    monkeypatch.setattr(
        AsyncPgVector,
        "similarity_search_with_score_by_vector",
        dummy_similarity_search_with_score_by_vector,
    )
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    # Override document addition functions.
    def dummy_add_documents(self, docs, ids):
        return ids

    async def dummy_aadd_documents(self, docs, ids=None, executor=None):
        return ids

    monkeypatch.setattr(AsyncPgVector, "add_documents", dummy_add_documents)
    monkeypatch.setattr(AsyncPgVector, "aadd_documents", dummy_aadd_documents)

    # Override delete function.
    async def dummy_delete(self, ids=None, collection_only=False, executor=None):
        return None

    async def dummy_delete_scoped(self, ids=None, owners=None, executor=None):
        return None

    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)
    monkeypatch.setattr(AsyncPgVector, "delete_scoped", dummy_delete_scoped)


def test_get_all_ids(auth_headers):
    response = client.get("/ids", headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert "testid1" in json_data


def test_get_documents_by_ids(auth_headers):
    response = client.get(
        "/documents", params={"ids": ["testid1"]}, headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert json_data[0]["page_content"] == "Test content"
    assert json_data[0]["metadata"]["file_id"] == "testid1"


def test_delete_documents(auth_headers):
    response = client.request(
        "DELETE", "/documents", json=["testid1"], headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "Documents for" in json_data["message"]


def test_query_embeddings_by_file_id(auth_headers):
    data = {
        "query": "Test query",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/query", json=data, headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_embed_local_file(tmp_path, auth_headers, monkeypatch):
    # Monkeypatch RAG_UPLOAD_DIR so the file is within the allowed directory.
    monkeypatch.setattr(document_routes, "RAG_UPLOAD_DIR", str(tmp_path))

    # Create a temporary file inside the patched upload dir.
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")

    data = {
        "filepath": "test.txt",
        "filename": "test.txt",
        "file_content_type": "text/plain",
        "file_id": "testid1",
    }
    response = client.post("/local/embed", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_embed_local_file_reports_storage_failure(
    tmp_path, auth_headers, monkeypatch
):
    monkeypatch.setattr(document_routes, "RAG_UPLOAD_DIR", str(tmp_path))
    test_file = tmp_path / "failed.txt"
    test_file.write_text("This document cannot be stored.")

    async def fail_storage(*args, **kwargs):
        return {"message": "storage failed", "error": "embedding failed"}

    monkeypatch.setattr(document_routes, "store_data_in_vector_db", fail_storage)
    response = client.post(
        "/local/embed",
        json={
            "filepath": "failed.txt",
            "filename": "failed.txt",
            "file_content_type": "text/plain",
            "file_id": "failed-id",
        },
        headers=auth_headers,
    )

    assert response.status_code == 500


def test_embed_file(tmp_path, auth_headers):
    file_content = "This is a test file for the embed endpoint."
    test_file = tmp_path / "test_embed.txt"
    test_file.write_text(file_content)
    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("test_embed.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_load_document_context(auth_headers):
    response = client.get("/documents/testid1/context", headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    content = response.text
    assert "testid1" in content or "Test content" in content


def test_load_document_context_restores_parallel_chunk_order(auth_headers, monkeypatch):
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    async def out_of_order_documents(self, ids, owners=None, executor=None):
        return [
            Document(
                page_content="third",
                metadata={
                    "_rag_chunk_index": 2,
                    "_rag_ingestion_attempt_id": "attempt-a",
                    "_rag_ingestion_attempt_started_at_ns": 100,
                },
            ),
            Document(
                page_content="first",
                metadata={
                    "_rag_chunk_index": 0,
                    "_rag_ingestion_attempt_id": "attempt-a",
                    "_rag_ingestion_attempt_started_at_ns": 100,
                },
            ),
            Document(
                page_content="second",
                metadata={
                    "_rag_chunk_index": 1,
                    "_rag_ingestion_attempt_id": "attempt-a",
                    "_rag_ingestion_attempt_started_at_ns": 100,
                },
            ),
        ]

    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", out_of_order_documents)

    response = client.get("/documents/testid1/context", headers=auth_headers)

    assert response.status_code == 200, f"Response: {response.text}"
    assert response.json() == "firstsecondthird"


def test_load_document_context_groups_repeated_ingestion_attempts(
    auth_headers, monkeypatch
):
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    def marked(content, chunk_index, attempt_id, started_at_ns):
        return Document(
            page_content=content,
            metadata={
                "_rag_chunk_index": chunk_index,
                "_rag_ingestion_attempt_id": attempt_id,
                "_rag_ingestion_attempt_started_at_ns": started_at_ns,
            },
        )

    async def interleaved_attempts(self, ids, owners=None, executor=None):
        return [
            marked("old-second", 1, "old", 100),
            marked("new-second", 1, "new", 200),
            marked("old-first", 0, "old", 100),
            marked("new-first", 0, "new", 200),
        ]

    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", interleaved_attempts)

    response = client.get("/documents/testid1/context", headers=auth_headers)

    assert response.status_code == 200, f"Response: {response.text}"
    assert response.json() == "old-firstold-secondnew-firstnew-second"


def test_embed_file_upload(tmp_path, auth_headers, monkeypatch):
    file_content = "Test content for embed upload."
    test_file = tmp_path / "upload_test.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed-upload",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"uploaded_file": ("upload_test.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_embed_file_upload_reports_storage_failure(
    tmp_path, auth_headers, monkeypatch
):
    test_file = tmp_path / "failed-upload.txt"
    test_file.write_text("This document cannot be stored.")

    async def fail_storage(*args, **kwargs):
        return {"message": "storage failed", "error": "embedding failed"}

    monkeypatch.setattr(document_routes, "store_data_in_vector_db", fail_storage)
    with test_file.open("rb") as file_handle:
        response = client.post(
            "/embed-upload",
            data={"file_id": "failed-id", "entity_id": "testuser"},
            files={
                "uploaded_file": (
                    "failed-upload.txt",
                    file_handle,
                    "text/plain",
                )
            },
            headers=auth_headers,
        )

    assert response.status_code == 500


def test_query_multiple(auth_headers):
    data = {
        "query": "Test query multiple",
        "file_ids": ["testid1", "testid2"],
        "k": 4,
    }
    response = client.post("/query_multiple", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_extract_text_from_file(tmp_path, auth_headers):
    """Test the /text endpoint for text extraction without embeddings."""
    file_content = "This is a test file for text extraction.\nIt has multiple lines.\nAnd should be extracted properly."
    test_file = tmp_path / "test_text_extraction.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/text",
            data={"file_id": "test_text_123", "entity_id": "testuser"},
            files={"file": ("test_text_extraction.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()

    # Check response structure
    assert "text" in json_data
    assert "file_id" in json_data
    assert "filename" in json_data
    assert "known_type" in json_data

    # Check response content
    assert json_data["text"] == file_content
    assert json_data["file_id"] == "test_text_123"
    assert json_data["filename"] == "test_text_extraction.txt"
    assert json_data["known_type"] is True  # text files are known types
