import asyncio
import json
from pathlib import Path

import pytest

from app.services.extraction_client import (
    ExtractionClient,
    ExtractionError,
    ExtractionSettings,
    INPUT_BYTES,
    TEXT_BYTES,
)


@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_raw_upload_and_valid_segments(processor, input_file, format):
    segments = await processor.client().extract(input_file, format)
    assert len(segments) == (2 if format == "pdf" else 1)
    assert [segment["index"] for segment in segments] == list(
        range(1, len(segments) + 1)
    )
    assert len(processor.requests) == 2
    method, path, headers, body = processor.requests[1]
    assert (method, path) == ("POST", "/v1/extract-text")
    assert body == Path(input_file).read_bytes()
    assert headers == {
        "Host": processor.url.removeprefix("http://"),
        "Authorization": f"Bearer {processor.token}",
        "Connection": "close",
        "Content-Length": str(len(body)),
        "Content-Type": "application/octet-stream",
        "X-Extraction-Version": "1",
        "X-Extraction-Format": format,
    }


@pytest.mark.parametrize("size", [0, INPUT_BYTES + 1])
async def test_input_bounds_before_any_connection(processor, input_file, size):
    with open(input_file, "wb") as file:
        file.truncate(size)
    with pytest.raises(ExtractionError, match="INPUT_LIMIT"):
        await processor.client().extract(input_file, "pdf")
    assert not processor.requests


@pytest.mark.parametrize(
    "change",
    [
        lambda r: {**r, "version": True},
        lambda r: {**r, "version": 2},
        lambda r: {**r, "operation": "execute"},
        lambda r: {**r, "format": "docx"},
        lambda r: {**r, "inputSha256": "0" * 64},
        lambda r: {**r, "textBytes": True},
        lambda r: {**r, "textBytes": r["textBytes"] + 1},
        lambda r: {**r, "textBytes": TEXT_BYTES + 1},
        lambda r: {**r, "metadata": {"user_id": "attacker"}},
        lambda r: {**r, "segments": []},
        lambda r: {**r, "segments": r["segments"] * 65},
        lambda r: {**r, "segments": list(reversed(r["segments"]))},
        lambda r: {**r, "segments": [{**r["segments"][0], "index": True}]},
        lambda r: {**r, "segments": [{**r["segments"][0], "index": 0}]},
        lambda r: {**r, "segments": [{**r["segments"][0], "kind": "document"}]},
        lambda r: {**r, "segments": [{**r["segments"][0], "text": "\ud800"}]},
        lambda r: {**r, "segments": [{**r["segments"][0], "text": None}]},
        lambda r: {**r, "segments": [{**r["segments"][0], "file_id": "forged"}]},
        lambda r: {
            **r,
            "textBytes": 1,
            "segments": [{"kind": "page", "index": 1, "text": " "}],
        },
    ],
)
async def test_strict_result_rejection(processor, input_file, change):
    processor.mutate = change
    with pytest.raises(ExtractionError, match="INVALID_RESPONSE"):
        await processor.client().extract(input_file, "pdf")


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(version=True),
        lambda c: c.update(operation="other"),
        lambda c: c.update(formats=["docx"]),
        lambda c: c.update(formats=["pdf", "pdf"]),
        lambda c: c.update(structure={"pdf": "pages"}),
        lambda c: c["limitations"].update(pdf="\ud800"),
        lambda c: c["limits"].update(inputBytes=True),
        lambda c: c["limits"].update(resultBytes=4 * 1024 * 1024),
        lambda c: c["limits"].update(concurrency=0),
        lambda c: c.update(extra="untrusted"),
    ],
)
async def test_incompatible_capabilities_never_upload(processor, input_file, change):
    change(processor.capabilities)
    with pytest.raises(ExtractionError, match="INCOMPATIBLE_SERVICE"):
        await processor.client().extract(input_file, "pdf")
    assert len(processor.requests) == 1


def response(body=b"{}", fields=None):
    fields = fields or f"Content-Length: {len(body)}\r\n".encode()
    return (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n"
        b"Connection: close\r\n" + fields + b"\r\n" + body
    )


@pytest.mark.parametrize(
    "raw",
    [
        response(b"\xff"),
        response(b'{"version":1,"version":1}'),
        response(b'{"textBytes":NaN}'),
        response(b"[]"),
        response(b"x" * 20, b"Content-Length: 21\r\n"),
        response(b"{}trailing", b"Content-Length: 2\r\n"),
        response(b"{}", b"Content-Length: 3145729\r\n"),
        response(b"{}", b"Content-Length: 2\r\nContent-Length: 2\r\n"),
        response(b"{}", b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n"),
        response(b"{}", b"Content-Length: 2\r\nContent-Encoding: gzip\r\n"),
        response(b"{}", b"Content-Length: 2\r\nX-Pad: " + b"x" * 4096 + b"\r\n"),
        b"HTTP/1.1 302 Redirect\r\nLocation: http://example.com\r\n\r\n",
    ],
)
async def test_bounded_strict_http_and_json(processor, input_file, raw):
    processor.raw = raw
    with pytest.raises(ExtractionError, match="INVALID_RESPONSE"):
        await processor.client().extract(input_file, "pdf")


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "INVALID_REQUEST"),
        (413, "INPUT_LIMIT"),
        (413, "STRUCTURE_LIMIT"),
        (422, "UNSUPPORTED_CONTENT"),
        (429, "BUSY"),
        (503, "UNAVAILABLE"),
        (504, "DEADLINE"),
    ],
)
async def test_safe_error_codes(processor, input_file, status, code):
    processor.error = (status, code)
    with pytest.raises(ExtractionError) as caught:
        await processor.client().extract(input_file, "pdf")
    assert (caught.value.code, caught.value.status_code) == (code, status)


async def test_deadline_closes_transport(processor, input_file):
    processor.hold = True
    client = processor.client(deadline=0.05)
    with pytest.raises(ExtractionError, match="DEADLINE"):
        await client.extract(input_file, "pdf")
    await asyncio.wait_for(processor.closed.wait(), 1)
    assert client._admitted == 0


async def test_cancellation_closes_transport_and_releases_admission(
    processor, input_file
):
    processor.hold = True
    client = processor.client()
    task = asyncio.create_task(client.extract(input_file, "pdf"))
    await asyncio.wait_for(processor.entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(processor.closed.wait(), 1)
    assert client._admitted == 0


async def test_client_admission_is_fail_fast(processor, input_file):
    processor.hold = True
    client = processor.client()
    tasks = [asyncio.create_task(client.extract(input_file, "pdf")) for _ in range(2)]
    while len([r for r in processor.requests if r[0] == "POST"]) < 2:
        await asyncio.sleep(0)
    try:
        with pytest.raises(ExtractionError, match="BUSY"):
            await client.extract(input_file, "pdf")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert client._admitted == 0


async def test_capabilities_rechecked_after_service_change(processor, input_file):
    client = processor.client()
    await client.extract(input_file, "pdf")
    processor.capabilities["version"] = 2
    with pytest.raises(ExtractionError, match="INCOMPATIBLE_SERVICE"):
        await client.extract(input_file, "pdf")
    assert [r[0] for r in processor.requests] == ["GET", "POST", "GET"]


async def test_outage_is_safe(input_file, processor, monkeypatch):
    async def unavailable(*args, **kwargs):
        raise ConnectionRefusedError("private endpoint")

    monkeypatch.setattr(asyncio, "open_connection", unavailable)
    client = processor.client()
    with pytest.raises(ExtractionError, match="UNAVAILABLE"):
        await client.extract(input_file, "pdf")


def test_default_settings_require_service_without_environment(monkeypatch):
    for key in (
        "RAG_EXTRACTION_ENABLED",
        "RAG_EXTRACTION_FORMATS",
        "RAG_EXTRACTION_SOCKET",
        "RAG_EXTRACTION_URL",
        "RAG_EXTRACTION_TOKEN_FILE",
        "RAG_EXTRACTION_DEADLINE_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = ExtractionSettings.from_env()
    assert settings.url == "http://extraction:8000" and settings.deadline == 10
    assert settings.token_file == "/run/extraction/token"
    assert not hasattr(settings, "enabled") and not hasattr(settings, "formats")


@pytest.mark.parametrize(
    "key,value",
    [
        ("RAG_EXTRACTION_SOCKET", "/socket/old.sock"),
        ("RAG_EXTRACTION_SOCKET", ""),
        ("RAG_EXTRACTION_ENABLED", "false"),
        ("RAG_EXTRACTION_ENABLED", "true"),
        ("RAG_EXTRACTION_ENABLED", ""),
        ("RAG_EXTRACTION_FORMATS", "docx"),
        ("RAG_EXTRACTION_FORMATS", "pdf,docx"),
        ("RAG_EXTRACTION_FORMATS", ""),
    ],
)
def test_obsolete_opt_out_configuration_is_rejected(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="obsolete"):
        ExtractionSettings.from_env()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_file": "relative"},
        {"token_file": ""},
        {"token_file": "/bad\x00path"},
        {"deadline": 0},
        {"deadline": 11},
        {"deadline": True},
        {"deadline": float("nan")},
        {"deadline": float("inf")},
    ],
)
def test_required_config_is_validated(kwargs):
    with pytest.raises(ValueError):
        ExtractionSettings(**kwargs)


@pytest.mark.parametrize(
    "key,value",
    [
        ("RAG_EXTRACTION_URL", "relative"),
        ("RAG_EXTRACTION_TOKEN_FILE", "relative"),
        ("RAG_EXTRACTION_DEADLINE_SECONDS", "stale"),
    ],
)
def test_invalid_environment_never_disables_extraction(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        ExtractionSettings.from_env()


@pytest.mark.parametrize(
    "kwargs", [{"enabled": False}, {"formats": frozenset({"docx"})}]
)
def test_client_constructor_cannot_opt_out(kwargs):
    with pytest.raises(TypeError):
        ExtractionSettings(**kwargs)
