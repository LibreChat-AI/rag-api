import asyncio
import socket
import ssl

import pytest

from app.services.extraction_client import (
    CHUNK_BYTES,
    INPUT_BYTES,
    ExtractionSettings,
    ExtractionClient,
    ExtractionError,
)


@pytest.mark.parametrize("https", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_backpressured_upload_abort_stops_input(
    processor, input_file, tls_certificate, monkeypatch, https, cancel
):
    with open(input_file, "wb") as file:
        file.truncate(INPUT_BYTES)
    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    received = 0
    tasks = set()

    async def handle(reader, writer):
        nonlocal received
        tasks.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            entered.set()
            await release.wait()
            while chunk := await reader.read(CHUNK_BYTES):
                received += len(chunk)
        except OSError:
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            if entered.is_set():
                closed.set()
            tasks.discard(asyncio.current_task())

    context = None
    if https:
        monkeypatch.setenv("SSL_CERT_FILE", str(tls_certificate[0]))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*tls_certificate)
    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=context)
    connect = asyncio.open_connection

    async def bounded_socket(*args, **kwargs):
        reader, writer = await connect(*args, **kwargs)
        writer.get_extra_info("socket").setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, 8192
        )
        return reader, writer

    monkeypatch.setattr(asyncio, "open_connection", bounded_socket)
    async with server:
        client = ExtractionClient(
            ExtractionSettings(
                url=f"{'https://localhost' if https else 'http://127.0.0.1'}:{server.sockets[0].getsockname()[1]}",
                token_file=processor.token_file,
                deadline=10 if cancel else 0.2,
            )
        )
        request = client._request

        async def extraction_only(method, *args, **kwargs):
            if method == "GET":
                return processor.capabilities, ""
            return await request(method, *args, **kwargs)

        monkeypatch.setattr(client, "_request", extraction_only)
        task = asyncio.create_task(client.extract(input_file, "pdf"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            if cancel:
                task.cancel()
            with pytest.raises(
                asyncio.CancelledError if cancel else ExtractionError
            ) as caught:
                await asyncio.wait_for(asyncio.shield(task), 1)
            if not cancel:
                assert caught.value.code == "DEADLINE"
            release.set()
            await asyncio.wait_for(closed.wait(), 1)
            assert received < INPUT_BYTES and client._admitted == 0
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            for active in list(tasks):
                active.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def test_repeated_cancellation_drains_aborted_writer(
    processor, input_file, monkeypatch
):
    processor.hold = True
    entered, release = asyncio.Event(), asyncio.Event()
    wait_closed = asyncio.StreamWriter.wait_closed

    async def delayed_close(writer):
        await wait_closed(writer)
        if processor.entered.is_set():
            entered.set()
            await release.wait()

    monkeypatch.setattr(asyncio.StreamWriter, "wait_closed", delayed_close)
    client = processor.client()
    task = asyncio.create_task(client.extract(input_file, "pdf"))
    try:
        await asyncio.wait_for(processor.entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(entered.wait(), 1)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done() and client._admitted == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        await asyncio.wait_for(processor.closed.wait(), 1)
        assert client._admitted == 0
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
