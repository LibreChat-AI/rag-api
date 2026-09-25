"""Contract tests for the opt-in extraction slice; uses the pinned native wheel."""

import asyncio
import io
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import jwt
import pytest
from fastapi import FastAPI, HTTPException, UploadFile

from app.middleware import security_middleware
from app.routes import extraction_routes
from app.services import extraction, extraction_worker
from main import app as main_app

app = FastAPI()
app.middleware("http")(security_middleware)
app.include_router(extraction_routes.router)

FIXTURE = Path(__file__).parent / "fixtures" / "structured.docx"
DOCX_TYPE = extraction_routes.DOCX_TYPE


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_EXTRACTION_API_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "a-test-key-that-is-at-least-32-bytes-long")
    monkeypatch.setattr(extraction_routes, "RAG_UPLOAD_DIR", str(tmp_path))
    admission = extraction.ExtractionAdmission()
    monkeypatch.setattr(extraction_routes, "_admission", admission)
    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(main_app.state, "thread_pool", pool, raising=False)
        yield tmp_path, admission


@pytest.fixture
def headers(configured):
    token = jwt.encode({"id": "owner"}, os.environ["JWT_SECRET"], algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


async def post(
    file_bytes, headers, name="report.docx", mime=DOCX_TYPE, profile="document-v1"
):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(
            "/v1/extract",
            headers=headers,
            data={"profile": profile},
            files={"file": (name, file_bytes, mime)},
        )


def with_entry(original: bytes, name: str, value: bytes) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(
        result, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for entry in source.infolist():
            if entry.filename != name:
                target.writestr(entry, source.read(entry))
        target.writestr(name, value)
    return result.getvalue()


async def test_real_docx_from_marcos_pr_returns_markdown_and_provenance(
    headers, configured
):
    response = await post(
        FIXTURE.read_bytes(), headers, name="renamed.csv", mime=DOCX_TYPE
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload == {
        "profile": "document-v1",
        "format": "markdown",
        "text": (
            "# Quarterly Report\n\nThis document summarizes the results for the period.\n\n"
            "## Regional Totals\n\n|  |  |  |\n| --- | --- | --- |\n"
            "| Region | Units | Revenue |\n| North | 1200 | 48000 |\n"
            "| South | 950 | 38000 |\n| East | 1430 | 57200 |\n\n"
            "**Totals are unaudited.**\n"
        ),
        "completeness": "complete",
        "may_omit_content": False,
        "pages_needing_ocr": [],
        "truncated": False,
        "parser": {"name": "anydoc", "version": "0.1.3"},
    }
    assert not list(configured[0].iterdir())


async def test_embedded_image_cannot_claim_complete_text(headers, configured):
    image_docx = with_entry(FIXTURE.read_bytes(), "word/media/scan.png", b"\x89PNG\r\n")
    response = await post(
        image_docx, headers, name="report.docx", mime="application/octet-stream"
    )
    assert response.status_code == 200, response.text
    assert response.json()["completeness"] == "partial"
    assert response.json()["may_omit_content"] is True
    assert response.json()["pages_needing_ocr"] == []
    assert not list(configured[0].iterdir())


def test_real_app_registers_route_only_when_enabled():
    # A fresh process verifies main.py registration, not just the test router.
    script = (
        "from langchain_community.vectorstores.pgvector import PGVector\n"
        "from app.services.vector_store.async_pg_vector import AsyncPgVector\n"
        "PGVector.__post_init__ = lambda self: None\n"
        "AsyncPgVector.__post_init__ = lambda self: None\n"
        "from main import app\n"
        "import sys\n"
        "print(int(any(getattr(r, 'path', None) == '/v1/extract' for r in app.routes)), "
        "int('anydoc' in sys.modules))\n"
    )
    for enabled, expected in (("false", "0 0"), ("true", "1 0")):
        env = {
            **os.environ,
            "RAG_EXTRACTION_API_ENABLED": enabled,
            "OPENAI_API_KEY": "test_key",
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip().splitlines()[-1] == expected, result.stderr


async def test_disabled_requires_explicit_opt_in(headers, configured, monkeypatch):
    monkeypatch.delenv("RAG_EXTRACTION_API_ENABLED")
    response = await post(FIXTURE.read_bytes(), headers)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "EXTRACTION_DISABLED"
    assert not list(configured[0].iterdir())


async def test_requires_verified_identity_and_signing_secret(
    headers, configured, monkeypatch
):
    missing = await post(FIXTURE.read_bytes(), {})
    assert missing.status_code == 401
    monkeypatch.delenv("JWT_SECRET")
    unsigned = await post(FIXTURE.read_bytes(), {})
    assert unsigned.status_code == 401
    assert unsigned.json()["detail"]["code"] == "EXTRACTION_AUTH_REQUIRED"
    assert not list(configured[0].iterdir())


@pytest.mark.parametrize(
    "name,mime,code",
    [
        ("report.md", "text/markdown", "UNSUPPORTED_DOCUMENT_TYPE"),
        ("report.docx", "application/pdf", "UNSUPPORTED_DOCUMENT_TYPE"),
        ("report.pdf", "application/octet-stream", "UNSUPPORTED_DOCUMENT_TYPE"),
    ],
)
async def test_unrelated_formats_never_reach_the_parser(
    headers, configured, name, mime, code
):
    response = await post(FIXTURE.read_bytes(), headers, name=name, mime=mime)
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == code
    assert not list(configured[0].iterdir())


async def test_nonfinite_deadline_cannot_disable_worker_timeout(
    headers, configured, monkeypatch
):
    monkeypatch.setenv("RAG_EXTRACTION_TIMEOUT_SECONDS", "inf")
    response = await post(FIXTURE.read_bytes(), headers)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "PARSER_UNAVAILABLE"
    assert not list(configured[0].iterdir())


async def test_unknown_profile_and_invalid_archive_fail_closed(headers, configured):
    bad_profile = await post(FIXTURE.read_bytes(), headers, profile="raw-v1")
    assert bad_profile.status_code == 400
    assert bad_profile.json()["detail"]["code"] == "UNSUPPORTED_PROFILE"
    bad_archive = await post(b"private secret from a forged DOCX", headers)
    assert bad_archive.status_code == 422
    assert bad_archive.json() == {"detail": {"code": "ARCHIVE_INVALID"}}
    assert not list(configured[0].iterdir())


async def test_refuses_zip_bomb_before_native_parsing(headers, configured):
    bomb = with_entry(
        FIXTURE.read_bytes(), "word/bomb.xml", b"x" * (25 * 1024 * 1024 + 1)
    )
    response = await post(bomb, headers)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "ZIP_BOMB"
    assert not list(configured[0].iterdir())


async def test_empty_docx_reports_no_text_instead_of_success(headers, configured):
    result = io.BytesIO()
    with zipfile.ZipFile(FIXTURE) as source, zipfile.ZipFile(
        result, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for entry in source.infolist():
            data = source.read(entry)
            if entry.filename == "word/document.xml":
                data = (
                    b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    b"<w:body><w:p/></w:body></w:document>"
                )
            target.writestr(entry, data)
    response = await post(result.getvalue(), headers)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "NO_DOCUMENT_TEXT"
    assert not list(configured[0].iterdir())


async def test_archive_entry_count_has_an_independent_limit(headers, configured):
    result = io.BytesIO()
    with zipfile.ZipFile(FIXTURE) as source, zipfile.ZipFile(
        result, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for entry in source.infolist():
            target.writestr(entry, source.read(entry))
        for index in range(4096):
            target.writestr(f"word/noise/{index}", b"")
    response = await post(result.getvalue(), headers)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "ZIP_BOMB"
    assert not list(configured[0].iterdir())


async def test_input_limit_is_checked_while_streaming_even_without_size_hint(
    configured, monkeypatch
):
    monkeypatch.setattr(extraction_routes, "MAX_INPUT_BYTES", 8)
    file = UploadFile(file=io.BytesIO(b"0123456789"), filename="report.docx", size=None)
    with pytest.raises(HTTPException) as caught:
        await extraction_routes._save_bounded(file, configured[0] / "bounded.docx")
    assert caught.value.status_code == 413
    assert caught.value.detail == {"code": "PARSER_INPUT_LIMIT"}


async def test_worker_output_limit_is_a_refusal(configured, monkeypatch):
    monkeypatch.setattr(extraction_worker, "MAX_OUTPUT_BYTES", 10)
    with pytest.raises(extraction_worker.ExtractionRefusal) as caught:
        extraction_worker.extract(FIXTURE)
    assert caught.value.code == "PARSER_OUTPUT_LIMIT"


async def test_child_crash_and_malformed_output_are_sanitized(
    headers, configured, monkeypatch
):
    monkeypatch.setattr(
        extraction,
        "_command",
        lambda path: (sys.executable, "-c", "import sys; sys.exit(11)"),
    )
    crash = await post(FIXTURE.read_bytes(), headers)
    assert crash.status_code == 503
    assert crash.json() == {"detail": {"code": "PARSER_CRASH"}}
    assert not list(configured[0].iterdir())
    monkeypatch.setattr(
        extraction,
        "_command",
        lambda path: (sys.executable, "-c", "print('private secret')"),
    )
    invalid = await post(FIXTURE.read_bytes(), headers)
    assert invalid.status_code == 503
    assert invalid.json() == {"detail": {"code": "PARSER_CRASH"}}
    assert "private secret" not in invalid.text
    assert not list(configured[0].iterdir())


async def test_api_kills_worker_that_overproduces_ipc(headers, configured, monkeypatch):
    monkeypatch.setattr(extraction, "MAX_IPC_BYTES", 64)
    monkeypatch.setattr(
        extraction,
        "_command",
        lambda path: (sys.executable, "-c", "import os; os.write(1, b'x' * 1024)"),
    )
    response = await post(FIXTURE.read_bytes(), headers)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "PARSER_OUTPUT_LIMIT"
    assert not list(configured[0].iterdir())
    assert configured[1]._pending == 0


def _sleeping_command(path: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-c",
        "import os,sys,time; open(sys.argv[1]+'.pid','w').write(str(os.getpid())); time.sleep(60)",
        str(path),
    )


async def _await_child(tmp_path: Path) -> Path:
    for _ in range(200):
        pids = list(tmp_path.glob("*.pid"))
        if pids:
            return pids[0]
        await asyncio.sleep(0.01)
    raise AssertionError("parser child did not start")


async def _assert_child_reaped(
    pid_file: Path, admission: extraction.ExtractionAdmission
) -> None:
    pid = int(pid_file.read_text())
    for _ in range(200):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            if admission._pending == 0 and not list(pid_file.parent.glob("*.docx")):
                pid_file.unlink()
                return
        await asyncio.sleep(0.01)
    raise AssertionError("parser child or staging file survived cancellation")


async def test_timeout_kills_and_reaps_child(headers, configured, monkeypatch):
    monkeypatch.setattr(extraction, "_command", _sleeping_command)
    monkeypatch.setenv("RAG_EXTRACTION_TIMEOUT_SECONDS", "0.3")
    task = asyncio.create_task(post(FIXTURE.read_bytes(), headers))
    pid_file = await _await_child(configured[0])
    response = await asyncio.wait_for(task, 2)
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "PARSER_TIMEOUT"
    await _assert_child_reaped(pid_file, configured[1])
    assert not list(configured[0].iterdir())
    assert configured[1]._pending == 0


async def test_cancel_reaps_child_and_frees_slot_for_next_upload(
    headers, configured, monkeypatch
):
    original_command = extraction._command
    monkeypatch.setattr(extraction, "_command", _sleeping_command)
    task = asyncio.create_task(post(FIXTURE.read_bytes(), headers))
    pid_file = await _await_child(configured[0])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    await _assert_child_reaped(pid_file, configured[1])
    assert not list(configured[0].iterdir())
    assert configured[1]._pending == 0
    monkeypatch.setattr(extraction, "_command", original_command)
    response = await post(FIXTURE.read_bytes(), headers)
    assert response.status_code == 200


async def test_busy_parser_refuses_before_staging_anything(
    headers, configured, monkeypatch
):
    monkeypatch.setattr(extraction, "_command", _sleeping_command)
    monkeypatch.setattr(
        extraction_routes, "_admission", extraction.ExtractionAdmission(1, 0)
    )
    task = asyncio.create_task(post(FIXTURE.read_bytes(), headers))
    pid_file = await _await_child(configured[0])
    busy = await post(FIXTURE.read_bytes(), headers)
    assert busy.status_code == 429
    assert busy.json()["detail"]["code"] == "CONCURRENCY_LIMIT"
    assert len(list(configured[0].glob("*.docx"))) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    await _assert_child_reaped(pid_file, extraction_routes._admission)
    assert not list(configured[0].iterdir())


async def test_existing_text_endpoint_still_preserves_raw_markdown(headers, configured):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/text",
            headers=headers,
            data={"file_id": "md-test"},
            files={"file": ("readme.md", b"# Raw **Markdown**\n", "text/markdown")},
        )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "# Raw **Markdown**"
