"""Route-level proof that the read and delete paths carry the caller's scope.

The store double here is not a stub: it evaluates the predicate the route builds
against real rows, so what these tests exercise is the route wiring — that every
path passes a scope at all, and that the scope it passes is the token's rather
than the caller's. The SQL that implements the predicate is covered against a
real database in ``tests/integration/test_authorization_pg.py``.
"""

import datetime
import os

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector
from main import app

JWT_SECRET = "testsecret"
VICTIM = "victim-user"
ATTACKER = "attacker-user"
AGENT = "agent-abc"

VICTIM_FILE = "file-victim"
ATTACKER_FILE = "file-attacker"
AGENT_FILE = "file-agent"

client = TestClient(app)


def _token(user_id: str) -> dict:
    os.environ["JWT_SECRET"] = JWT_SECRET
    payload = {
        "id": user_id,
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    return {
        "Authorization": f"Bearer {jwt.encode(payload, JWT_SECRET, algorithm='HS256')}"
    }


@pytest.fixture
def attacker_headers():
    return _token(ATTACKER)


@pytest.fixture
def victim_headers():
    return _token(VICTIM)


SHARED_FILE = "file-shared-id"

ROWS = [
    {"file_id": VICTIM_FILE, "user_id": VICTIM, "text": "victim secret"},
    {"file_id": ATTACKER_FILE, "user_id": ATTACKER, "text": "attacker note"},
    {"file_id": AGENT_FILE, "user_id": AGENT, "text": "agent knowledge"},
    # Two owners under one caller-chosen file id. The attacker's row is listed
    # first so it lands at documents[0] in an unfiltered search.
    {"file_id": SHARED_FILE, "user_id": ATTACKER, "text": "attacker own row"},
    {"file_id": SHARED_FILE, "user_id": VICTIM, "text": "victim hidden row"},
]


def _owners_from_filter(query_filter: dict) -> list:
    """Read the owner set out of the predicate the route built.

    Raises when the predicate carries no owner clause: a route that forgot to
    scope its query must fail these tests rather than quietly return everything.
    """
    clauses = query_filter.get("$and")
    assert clauses, f"route built an unscoped filter: {query_filter!r}"
    for clause in clauses:
        if "user_id" in clause:
            return clause["user_id"]["$in"]
    raise AssertionError(f"predicate carries no owner clause: {query_filter!r}")


def _files_from_filter(query_filter: dict) -> list:
    for clause in query_filter["$and"]:
        if "file_id" in clause:
            spec = clause["file_id"]
            return [spec["$eq"]] if "$eq" in spec else spec["$in"]
    raise AssertionError(f"predicate carries no file clause: {query_filter!r}")


def _matching(query_filter: dict) -> list:
    owners = _owners_from_filter(query_filter)
    files = _files_from_filter(query_filter)
    return [row for row in ROWS if row["user_id"] in owners and row["file_id"] in files]


@pytest.fixture(autouse=True)
def store_double(monkeypatch):
    """Patch AsyncPgVector so every read honours the predicate it is given."""
    document_routes.get_cached_query_embedding.cache_clear()
    monkeypatch.setattr(
        document_routes, "get_cached_query_embedding", lambda query: [0.1, 0.2, 0.3]
    )

    if getattr(app.state, "thread_pool", None) is None:
        from concurrent.futures import ThreadPoolExecutor

        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    deleted = []

    async def get_all_ids(self, owners, executor=None):
        return [row["file_id"] for row in ROWS if row["user_id"] in owners]

    async def get_filtered_ids(self, ids, owners, executor=None):
        return [
            row["file_id"]
            for row in ROWS
            if row["user_id"] in owners and row["file_id"] in ids
        ]

    async def get_documents_by_ids(self, ids, owners, executor=None):
        return [
            Document(
                page_content=row["text"],
                metadata={"file_id": row["file_id"], "user_id": row["user_id"]},
            )
            for row in ROWS
            if row["user_id"] in owners and row["file_id"] in ids
        ]

    async def delete_scoped(self, ids, owners, executor=None):
        deleted.extend(
            row["file_id"]
            for row in ROWS
            if row["user_id"] in owners and row["file_id"] in ids
        )

    async def asimilarity_search(self, embedding, k=4, filter=None, executor=None):
        return [
            (
                Document(
                    page_content=row["text"],
                    metadata={"file_id": row["file_id"], "user_id": row["user_id"]},
                ),
                0.1,
            )
            for row in _matching(filter)
        ]

    monkeypatch.setattr(AsyncPgVector, "get_all_ids", get_all_ids)
    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", get_filtered_ids)
    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", get_documents_by_ids)
    monkeypatch.setattr(AsyncPgVector, "delete_scoped", delete_scoped)
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        asimilarity_search,
    )
    monkeypatch.setattr(document_routes, "_apply_distance_threshold", lambda docs: docs)
    return deleted


def test_ids_lists_only_the_callers_files(attacker_headers):
    """``GET /ids`` returned every file id in the deployment, which is the
    discovery half of the read chain."""
    response = client.get("/ids", headers=attacker_headers)

    assert response.status_code == 200
    assert set(response.json()) == {ATTACKER_FILE, SHARED_FILE}
    assert VICTIM_FILE not in response.json()
    assert AGENT_FILE not in response.json()


def test_query_multiple_no_longer_returns_foreign_files(attacker_headers):
    """``/query_multiple`` performed no authorization at all: every file id the
    caller listed was searched and returned. Paired with ``GET /ids`` above,
    that disclosed the content of the whole deployment to any authenticated
    caller."""
    response = client.post(
        "/query_multiple",
        json={
            "query": "anything",
            "file_ids": [VICTIM_FILE, ATTACKER_FILE, AGENT_FILE],
            "k": 10,
        },
        headers=attacker_headers,
    )

    assert response.status_code == 200
    contents = [entry[0]["page_content"] for entry in response.json()]
    assert contents == ["attacker note"]


def test_query_authorizes_every_hit_not_just_the_first(attacker_headers):
    """``/query`` authorized the whole result set from ``documents[0]`` — the
    first *returned* hit — so any hit behind it was never checked.

    A file id is caller-chosen, so an attacker could upload one row under a
    victim's file id, have their own row rank first, and read the victim's rows
    that followed. The predicate now runs before ranking, so the foreign rows are
    never in the result set to begin with.
    """
    response = client.post(
        "/query",
        json={"query": "anything", "file_id": SHARED_FILE, "k": 10},
        headers=attacker_headers,
    )

    assert response.status_code == 200
    contents = [entry[0]["page_content"] for entry in response.json()]
    assert contents == ["attacker own row"]


def test_query_without_entity_id_cannot_reach_another_owner(attacker_headers):
    """The caller's own identity is the whole scope when no entity is named, so
    a victim's file id returns nothing rather than the victim's chunks."""
    response = client.post(
        "/query",
        json={"query": "anything", "file_id": VICTIM_FILE, "k": 10},
        headers=attacker_headers,
    )

    assert response.status_code == 200
    assert response.json() == []


def test_entity_id_is_still_caller_asserted(attacker_headers):
    """Known residual, carried deliberately by this release.

    ``entity_id`` no longer *replaces* the caller's identity — it widens the
    owner set, and the caller's own identity always stays in it. But nothing in
    a token minted today proves the caller may act for the entity it names, so a
    caller who knows a victim's user id can still name it and reach that owner.

    Closing this requires the token to carry the entity authorization, which is a
    coordinated change with the callers that mint those tokens. It is tracked
    separately from this release; see ``README.md``. This test asserts the gap so
    that closing it is a deliberate edit rather than a silent behaviour change.
    """
    response = client.post(
        "/query",
        json={
            "query": "anything",
            "file_id": VICTIM_FILE,
            "k": 10,
            "entity_id": VICTIM,
        },
        headers=attacker_headers,
    )

    assert response.status_code == 200
    contents = [entry[0]["page_content"] for entry in response.json()]
    assert contents == ["victim secret"], (
        "entity_id reach changed — if a token now proves entity authorization, "
        "this test should assert refusal instead"
    )


def test_query_reaches_an_agent_knowledge_base(attacker_headers):
    """Agent knowledge-base chunks are owned by the agent id, so a caller
    reading one must reach both owners. This is the behaviour ``entity_id``
    exists for and it must keep working."""
    response = client.post(
        "/query",
        json={
            "query": "anything",
            "file_id": AGENT_FILE,
            "k": 10,
            "entity_id": AGENT,
        },
        headers=attacker_headers,
    )

    assert response.status_code == 200
    contents = [entry[0]["page_content"] for entry in response.json()]
    assert contents == ["agent knowledge"]


def test_documents_route_refuses_a_foreign_file(attacker_headers):
    """``GET /documents`` returned the chunks of any file id the caller named."""
    response = client.get(
        "/documents", params={"ids": [VICTIM_FILE]}, headers=attacker_headers
    )

    assert response.status_code == 404


def test_document_context_refuses_a_foreign_file(attacker_headers):
    """``/documents/{id}/context`` had the same reach as ``GET /documents``."""
    response = client.get(f"/documents/{VICTIM_FILE}/context", headers=attacker_headers)

    assert response.status_code == 404


def test_delete_cannot_remove_a_foreign_file(attacker_headers, store_double):
    """``DELETE /documents`` deleted by file id alone, so any authenticated
    caller could destroy another owner's chunks by naming one."""
    response = client.request(
        "DELETE", "/documents", json=[VICTIM_FILE], headers=attacker_headers
    )

    assert response.status_code == 404
    assert store_double == []


def test_delete_accepts_entity_id_as_a_query_parameter(attacker_headers, store_double):
    """Entity-owned chunks must be deletable by naming the entity.

    Chunks embedded under an ``entity_id`` — an agent knowledge base — are owned
    by that entity rather than by any user, so a delete scoped to the caller
    alone matches nothing and orphans them. This route takes ``entity_id`` on the
    query string alongside the JSON body of file ids, which is the shape callers
    send; a delete that only accepted it in the body would silently drop it and
    orphan every agent knowledge-base file.
    """
    response = client.request(
        "DELETE",
        "/documents",
        params={"entity_id": AGENT},
        json=[AGENT_FILE],
        headers=attacker_headers,
    )

    assert response.status_code == 200
    assert store_double == [AGENT_FILE]


def test_delete_without_entity_id_leaves_entity_owned_chunks(
    attacker_headers, store_double
):
    """The failure mode the query parameter above exists to prevent.

    Pinned deliberately: this is what a caller that has not been updated will
    do, and the 404 it gets is indistinguishable from "already deleted" — so the
    orphan is silent unless the caller names the entity.
    """
    response = client.request(
        "DELETE", "/documents", json=[AGENT_FILE], headers=attacker_headers
    )

    assert response.status_code == 404
    assert store_double == []


def test_partial_delete_destroys_nothing_before_reporting_not_found(
    attacker_headers, store_double
):
    """A delete mixing an owned id with an unknown one must remove neither.

    The existence check has to run *before* the delete. With the order reversed
    the owned rows are destroyed and the caller is still told "not found", so a
    client that treats 404 as "nothing happened" — which is exactly how the 404
    reads — silently loses data it never asked to delete. Reported against the
    first version of this route and re-pinned here.
    """
    response = client.request(
        "DELETE",
        "/documents",
        json=[ATTACKER_FILE, "file-does-not-exist"],
        headers=attacker_headers,
    )

    assert response.status_code == 404
    assert store_double == []


def test_owner_still_reads_and_deletes_their_own(victim_headers, store_double):
    """The scope must not lock owners out of their own content."""
    read = client.get(
        "/documents", params={"ids": [VICTIM_FILE]}, headers=victim_headers
    )
    assert read.status_code == 200
    assert read.json()[0]["page_content"] == "victim secret"

    removed = client.request(
        "DELETE", "/documents", json=[VICTIM_FILE], headers=victim_headers
    )
    assert removed.status_code == 200
    assert store_double == [VICTIM_FILE]


def test_unauthenticated_deployment_keeps_reading_its_own_chunks(monkeypatch):
    """A deployment with no ``JWT_SECRET`` has no token to resolve scope from.

    Writes stamp ``public`` as the owner in that mode, so reads must resolve to
    the same single owner rather than to an empty scope — otherwise upgrading
    would lock these deployments out of everything they hold.
    """
    monkeypatch.delenv("JWT_SECRET", raising=False)
    public_file = "file-public"
    ROWS.append({"file_id": public_file, "user_id": "public", "text": "public row"})

    try:
        response = client.post(
            "/query",
            json={"query": "anything", "file_id": public_file, "k": 10},
        )

        assert response.status_code == 200
        contents = [entry[0]["page_content"] for entry in response.json()]
        assert contents == ["public row"]

        listed = client.get("/ids")
        assert listed.status_code == 200
        assert listed.json() == [public_file]
    finally:
        ROWS.pop()


def test_exists_answers_for_the_named_entity_alone(attacker_headers):
    """A presence check names one namespace. The caller's own row under the
    shared file id must not answer for the agent, which ``/query`` cannot tell
    apart because it reads the caller and the entity together."""
    response = client.post(
        "/exists",
        json={"file_ids": [SHARED_FILE, AGENT_FILE], "entity_id": AGENT},
        headers=attacker_headers,
    )

    assert response.status_code == 200
    assert response.json() == {"file_ids": [AGENT_FILE]}


def test_exists_without_entity_id_answers_for_the_caller(attacker_headers):
    """With no entity named the caller is the namespace, so a victim's file id
    reads as absent and the route is not an existence oracle."""
    response = client.post(
        "/exists",
        json={"file_ids": [VICTIM_FILE, SHARED_FILE, ATTACKER_FILE, SHARED_FILE]},
        headers=attacker_headers,
    )

    assert response.status_code == 200
    assert response.json() == {"file_ids": [SHARED_FILE, ATTACKER_FILE]}


def test_exists_on_an_unauthenticated_deployment(monkeypatch):
    """Without ``JWT_SECRET`` writes stamp the entity or ``public``, and the
    presence check resolves to the same owner."""
    monkeypatch.delenv("JWT_SECRET", raising=False)

    named = client.post(
        "/exists", json={"file_ids": [AGENT_FILE, VICTIM_FILE], "entity_id": AGENT}
    )
    unnamed = client.post("/exists", json={"file_ids": [AGENT_FILE]})

    assert named.json() == {"file_ids": [AGENT_FILE]}
    assert unnamed.json() == {"file_ids": []}
