import asyncio
import os
import ssl
import threading
from pathlib import Path

import pytest

from app.services import extraction_client
from app.services.extraction_client import ExtractionError, ExtractionSettings


@pytest.mark.parametrize(
    "url",
    [
        "http://extraction:8000",
        "http://extraction:8000/",
        "http://localhost:1234",
        "http://127.0.0.1:1234",
        "http://[::1]:1234",
        "https://processor.example",
        "https://processor.example:8443/",
        "https://[2001:db8::1]:8443",
    ],
)
def test_valid_origins(url):
    assert ExtractionSettings(url=url).url == url


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "extraction:8000",
        "unix:///socket/extraction.sock",
        "file:///run/extraction/token",
        "ftp://extraction:8000",
        "http://extraction",
        "http://extraction:8001",
        "http://remote.example:8000",
        "http://10.0.0.1:8000",
        "http://192.168.1.2:8000",
        "http://127.0.0.2:8000",
        "http://localhost.evil:8000",
        "https://user:secret@processor.example",
        "https://secret@processor.example",
        "https://processor.example/path",
        "https://processor.example//",
        "https://processor.example?token=secret",
        "https://processor.example?",
        "https://processor.example#secret",
        "https://processor.example#",
        "https://processor.example:0",
        "https://processor.example:65536",
        "https://processor.example:secret",
        "https://processor.example:",
        "https://[broken",
        "https://[::1]ignored",
        "https://[::1]:0443",
        "https://[v1.localhost]",
        "https://processor.example:0443",
        "https://processor.example\\@localhost",
        " https://processor.example",
        "https://processor.example\n",
        "https://processor.example\x00",
        "https://processor.example\t:443",
        "https://procéssor.example",
    ],
)
def test_invalid_origin_is_safe(url):
    with pytest.raises(ValueError) as caught:
        ExtractionSettings(url=url)
    assert "secret" not in str(caught.value)
    assert "processor.example" not in str(caught.value)


@pytest.mark.parametrize("phase", ["capabilities", "extract"])
async def test_both_operations_require_service_auth(processor, input_file, phase):
    if phase == "capabilities":
        Path(processor.token_file).write_text("cd" * 32)
    else:
        processor.error = (401, "UNAUTHORIZED")
    with pytest.raises(ExtractionError) as caught:
        await processor.client().extract(input_file, "pdf")
    assert (caught.value.code, caught.value.status_code) == ("UNAVAILABLE", 503)
    assert len(processor.requests) == (1 if phase == "capabilities" else 2)


@pytest.mark.parametrize(
    "token",
    [b"", b"a" * 63, b"a" * 65, b"a" * 64 + b"\n", b"A" * 64, b"g" * 64, b"\xff" * 64],
)
async def test_invalid_credentials_never_connect(processor, input_file, token):
    Path(processor.token_file).write_bytes(token)
    with pytest.raises(ExtractionError) as caught:
        await processor.client().extract(input_file, "pdf")
    assert (caught.value.code, caught.value.status_code) == ("UNAVAILABLE", 503)
    assert not processor.requests


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "fifo", "symlink", "oversized"]
)
async def test_nonregular_or_missing_credentials_fail_closed(
    processor, input_file, kind
):
    path = Path(processor.token_file)
    path.unlink()
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        target = path.with_name("other-token")
        target.write_text(processor.token)
        path.symlink_to(target)
    elif kind == "oversized":
        with path.open("wb") as file:
            file.truncate(1024 * 1024 * 1024)
    with pytest.raises(ExtractionError, match="UNAVAILABLE"):
        await asyncio.wait_for(processor.client().extract(input_file, "pdf"), 1)
    assert not processor.requests


async def test_token_read_is_bounded_nonblocking_and_once_per_job(
    processor, input_file, monkeypatch
):
    original_open, original_read = os.open, os.read
    opens, reads = [], []

    def capture_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if str(path) == processor.token_file:
            opens.append((descriptor, flags))
        return descriptor

    def capture_read(descriptor, size):
        if opens and descriptor == opens[-1][0]:
            reads.append(size)
        return original_read(descriptor, size)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "read", capture_read)
    await processor.client().extract(input_file, "pdf")
    assert len(opens) == 1 and reads == [65]
    assert opens[0][1] & os.O_NONBLOCK and opens[0][1] & os.O_NOFOLLOW
    with pytest.raises(OSError):
        os.fstat(opens[0][0])


async def test_token_rotation_is_visible_on_next_job(processor, input_file):
    client = processor.client()
    await client.extract(input_file, "pdf")
    replacement = Path(processor.token_file).with_name("rotated-token")
    processor.token = "cd" * 32
    replacement.write_text(processor.token)
    replacement.replace(processor.token_file)
    await client.extract(input_file, "pdf")
    assert [request[2]["Authorization"] for request in processor.requests] == [
        "Bearer " + "ab" * 32,
        "Bearer " + "ab" * 32,
        "Bearer " + "cd" * 32,
        "Bearer " + "cd" * 32,
    ]


async def test_proxy_environment_is_not_used(processor, input_file, monkeypatch):
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(key, "http://secret@127.0.0.1:1")
    await processor.client().extract(input_file, "pdf")
    assert len(processor.requests) == 2


async def test_credential_read_is_inside_deadline(processor, input_file, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    read = extraction_client._read_token

    def delayed_read(path):
        entered.set()
        try:
            release.wait(2)
            return read(path)
        finally:
            finished.set()

    monkeypatch.setattr(extraction_client, "_read_token", delayed_read)
    client = processor.client(deadline=0.05)
    try:
        with pytest.raises(ExtractionError, match="DEADLINE"):
            await client.extract(input_file, "pdf")
        assert entered.is_set() and not processor.requests and client._admitted == 0
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 1)


async def test_connect_is_inside_deadline(processor, input_file, monkeypatch):
    entered, closed = asyncio.Event(), asyncio.Event()

    async def delayed_connect(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            closed.set()

    monkeypatch.setattr(asyncio, "open_connection", delayed_connect)
    with pytest.raises(ExtractionError, match="DEADLINE"):
        await processor.client(deadline=0.05).extract(input_file, "pdf")
    assert entered.is_set() and closed.is_set() and not processor.requests


async def test_capabilities_are_inside_deadline(processor, input_file):
    processor.hold_caps = True
    with pytest.raises(ExtractionError, match="DEADLINE"):
        await processor.client(deadline=0.05).extract(input_file, "pdf")
    await asyncio.wait_for(processor.closed.wait(), 1)
    assert processor.caps_entered.is_set() and len(processor.requests) == 1


@pytest.fixture
async def tls_processor(processor, tls_certificate):
    certificate, key = tls_certificate
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    server = await asyncio.start_server(processor.handle, "127.0.0.1", 0, ssl=context)
    processor.url = f"https://localhost:{server.sockets[0].getsockname()[1]}"
    async with server:
        yield processor


async def test_https_uses_standard_trust_and_hostname(
    tls_processor, tls_certificate, input_file, monkeypatch
):
    monkeypatch.setenv("SSL_CERT_FILE", str(tls_certificate[0]))
    client = tls_processor.client()
    assert client._ssl.verify_mode == ssl.CERT_REQUIRED and client._ssl.check_hostname
    await client.extract(input_file, "pdf")
    assert len(tls_processor.requests) == 2


@pytest.mark.parametrize("failure", ["untrusted", "hostname"])
async def test_https_never_skips_verification(
    tls_processor, tls_certificate, input_file, monkeypatch, failure
):
    if failure == "hostname":
        monkeypatch.setenv("SSL_CERT_FILE", str(tls_certificate[0]))
        tls_processor.url = tls_processor.url.replace("localhost", "127.0.0.1")
    with pytest.raises(ExtractionError) as caught:
        await tls_processor.client().extract(input_file, "pdf")
    assert (caught.value.code, caught.value.status_code) == ("UNAVAILABLE", 503)
    assert not tls_processor.requests


@pytest.mark.parametrize("cancel", [False, True])
async def test_https_abort_closes_real_transport(
    tls_processor, tls_certificate, input_file, monkeypatch, cancel
):
    monkeypatch.setenv("SSL_CERT_FILE", str(tls_certificate[0]))
    tls_processor.hold = True
    client = tls_processor.client(deadline=0.1 if not cancel else 10)
    task = asyncio.create_task(client.extract(input_file, "pdf"))
    await asyncio.wait_for(tls_processor.entered.wait(), 1)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else ExtractionError):
        await task
    await asyncio.wait_for(tls_processor.closed.wait(), 1)
    assert client._admitted == 0


async def test_tls_handshake_deadline_aborts_connection(processor, input_file):
    connected, closed = asyncio.Event(), asyncio.Event()

    async def never_tls(reader, writer):
        connected.set()
        try:
            await reader.read()
            closed.set()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(never_tls, "127.0.0.1", 0)
    async with server:
        processor.url = f"https://localhost:{server.sockets[0].getsockname()[1]}"
        with pytest.raises(ExtractionError, match="DEADLINE"):
            await processor.client(deadline=0.05).extract(input_file, "pdf")
        assert connected.is_set()
        await asyncio.wait_for(closed.wait(), 1)
    assert not processor.requests


async def test_capabilities_and_result_share_one_deadline(
    processor, input_file, monkeypatch
):
    client = processor.client(deadline=0.05)
    request = client._request
    remaining = []

    async def delay(method, *args, **kwargs):
        if method == "POST":
            remaining.append(asyncio.current_task())
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0.03)
        return await request(method, *args, **kwargs)

    monkeypatch.setattr(client, "_request", delay)
    with pytest.raises(ExtractionError, match="DEADLINE"):
        await client.extract(input_file, "pdf")
    assert remaining and len(processor.requests) == 1
    assert client._admitted == 0


async def test_remote_error_text_and_redirects_are_not_reflected(processor, input_file):
    secret = b"private-service-details"
    body = b'{"message":"private-service-details"}'
    processor.raw = (
        b"HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json; charset=utf-8\r\n"
        + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        + body
    )
    with pytest.raises(ExtractionError) as caught:
        await processor.client().extract(input_file, "pdf")
    assert secret.decode() not in str(caught.value)
    processor.raw = (
        b"HTTP/1.1 302 Redirect\r\nLocation: https://secret@private.example/\r\n"
        b"Content-Type: application/json; charset=utf-8\r\nContent-Length: 2\r\n"
        b"Connection: close\r\n\r\n{}"
    )
    with pytest.raises(ExtractionError, match="INVALID_RESPONSE"):
        await processor.client().extract(input_file, "pdf")
    assert len(processor.requests) == 4


def test_invalid_deadline_environment_does_not_echo_value(monkeypatch):
    monkeypatch.setenv("RAG_EXTRACTION_DEADLINE_SECONDS", "private-service-details")
    with pytest.raises(ValueError) as caught:
        ExtractionSettings.from_env()
    assert "private-service-details" not in str(caught.value)


@pytest.mark.parametrize(
    "url", ["https://processor.example:8443", "http://localhost:2345"]
)
def test_service_environment_uses_only_service_configuration(monkeypatch, url):
    monkeypatch.setenv("RAG_EXTRACTION_URL", url)
    monkeypatch.setenv("RAG_EXTRACTION_TOKEN_FILE", "/run/extraction/custom-token")
    monkeypatch.setenv("RAG_EXTRACTION_DEADLINE_SECONDS", "4.5")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-app-credential")
    monkeypatch.setenv("JWT_SECRET", "unrelated-user-auth-secret")
    settings = ExtractionSettings.from_env()
    assert settings.url == url
    assert settings.token_file == "/run/extraction/custom-token"
    assert settings.deadline == 4.5
