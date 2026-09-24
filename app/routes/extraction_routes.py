"""Versioned document extraction. No embeddings, vector writes, or OCR calls."""

import asyncio
import math
import os
import tempfile
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.config import RAG_UPLOAD_DIR, logger
from app.models import ExtractionResult
from app.services.extraction import (
    ExtractionAdmission,
    ExtractionBusy,
    ExtractionFailure,
    run_worker,
)

router = APIRouter(prefix="/v1")
DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MAX_INPUT_BYTES = 15 * 1024 * 1024
_DEFAULT_TIMEOUT = 30.0
_admission: ExtractionAdmission | None = None


def _error(code: str, status_code: int) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code})


def _get_admission() -> ExtractionAdmission:
    global _admission
    if _admission is None:
        _admission = ExtractionAdmission(
            concurrent=int(os.getenv("RAG_EXTRACTION_CONCURRENT", "2")),
            queued=int(os.getenv("RAG_EXTRACTION_QUEUED", "6")),
        )
    return _admission


async def _save_bounded(file: UploadFile, path: Path) -> None:
    size = 0
    async with aiofiles.open(path, "wb") as output:
        while chunk := await file.read(64 * 1024):
            size += len(chunk)
            if size > MAX_INPUT_BYTES:
                raise _error("PARSER_INPUT_LIMIT", 413)
            await output.write(chunk)


@router.post("/extract", response_model=ExtractionResult)
async def extract_document(
    request: Request,
    file: UploadFile = File(...),
    profile: str = Form(...),
) -> ExtractionResult:
    # Existing /text remains unchanged. An operator must explicitly enable
    # and install this separate profile before moving any LibreChat caller.
    if os.getenv("RAG_EXTRACTION_API_ENABLED", "false").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise _error("EXTRACTION_DISABLED", 404)
    # Legacy RAG deployments may run without auth; expensive extraction is
    # never allowed anonymously, even when those older routes are public.
    if not os.getenv("JWT_SECRET") or not getattr(request.state, "user", {}).get("id"):
        raise _error("EXTRACTION_AUTH_REQUIRED", 401)
    if profile != "document-v1":
        raise _error("UNSUPPORTED_PROFILE", 400)
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    extension = Path(file.filename or "").suffix.lower()
    if content_type == "application/pdf" or not (
        content_type == DOCX_TYPE
        or (
            content_type in {"application/octet-stream", "binary/octet-stream", ""}
            and extension == ".docx"
        )
    ):
        raise _error("UNSUPPORTED_DOCUMENT_TYPE", 415)
    if file.size is not None and file.size > MAX_INPUT_BYTES:
        raise _error("PARSER_INPUT_LIMIT", 413)

    try:
        admission = _get_admission()
        timeout = float(
            os.getenv("RAG_EXTRACTION_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT))
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid extraction timeout")
        async with asyncio.timeout(timeout):
            async with admission.slot():
                fd, filename = tempfile.mkstemp(
                    prefix="rag-extract-", suffix=".docx", dir=RAG_UPLOAD_DIR
                )
                os.close(fd)
                path = Path(filename)
                try:
                    await _save_bounded(file, path)
                    return await run_worker(path)
                finally:
                    path.unlink(missing_ok=True)
    except ExtractionBusy:
        raise _error("CONCURRENCY_LIMIT", 429)
    except ExtractionFailure as exc:
        status_code = {
            "ZIP_BOMB": 413,
            "ARCHIVE_INVALID": 422,
            "PARSER_OUTPUT_LIMIT": 413,
            "NO_DOCUMENT_TEXT": 422,
            "PARSE_FAILED": 422,
            "PARSER_UNAVAILABLE": 503,
            "PARSER_CRASH": 503,
        }[exc.code]
        raise _error(exc.code, status_code)
    except TimeoutError:
        raise _error("PARSER_TIMEOUT", 504)
    except (OSError, ValueError) as exc:
        logger.error(
            "Extraction infrastructure unavailable | error=%s", type(exc).__name__
        )
        raise _error("PARSER_UNAVAILABLE", 503)
