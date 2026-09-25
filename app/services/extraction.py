"""Bounded, cancellable process boundary for opt-in document extraction."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from app.models import ExtractionResult

MAX_IPC_BYTES = 15 * 1024 * 1024


class ExtractionBusy(Exception):
    pass


class ExtractionFailure(Exception):
    def __init__(self, code: str):
        self.code = code


class ExtractionAdmission:
    """One per serving process: bound uploads waiting and native children running."""

    def __init__(self, concurrent: int = 2, queued: int = 6):
        if concurrent < 1 or queued < 0:
            raise ValueError("Invalid extraction admission limits")
        self._capacity = concurrent + queued
        self._pending = 0
        self._slots = asyncio.Semaphore(concurrent)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        # The serving process has one event loop. No await separates the check
        # and increment, so concurrent requests cannot exceed the queue limit.
        if self._pending >= self._capacity:
            raise ExtractionBusy()
        self._pending += 1
        acquired = False
        try:
            await self._slots.acquire()
            acquired = True
            yield
        finally:
            self._pending -= 1
            if acquired:
                self._slots.release()


def _command(path: Path) -> tuple[str, ...]:
    return (sys.executable, "-m", "app.services.extraction_worker", str(path))


async def run_worker(path: Path) -> ExtractionResult:
    """Read a bounded child response; always reap a child before releasing its slot."""
    process = await asyncio.create_subprocess_exec(
        *_command(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    try:
        output = bytearray()
        while chunk := await process.stdout.read(64 * 1024):
            output.extend(chunk)
            if len(output) > MAX_IPC_BYTES:
                raise ExtractionFailure("PARSER_OUTPUT_LIMIT")
        await process.wait()
    except BaseException:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass  # The child exited while cancellation was being delivered.
        # Draining and reaping are necessary before the temporary file can be
        # removed and admission can be granted to the next upload.
        await process.communicate()
        raise
    if process.returncode != 0:
        raise ExtractionFailure("PARSER_CRASH")
    try:
        message = json.loads(output)
        if message.get("ok") is False:
            code = message["code"]
            if code in {
                "ZIP_BOMB",
                "ARCHIVE_INVALID",
                "PARSER_UNAVAILABLE",
                "PARSER_OUTPUT_LIMIT",
                "NO_DOCUMENT_TEXT",
                "PARSE_FAILED",
            }:
                raise ExtractionFailure(code)
        return ExtractionResult.model_validate(message["result"])
    except ExtractionFailure:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExtractionFailure("PARSER_CRASH") from exc
