# app/utils/document_loader.py

import os
import codecs
import tempfile

from typing import Iterator, List, Optional
import chardet

from langchain_core.documents import Document

from app.config import (
    known_source_ext,
    PDF_EXTRACT_IMAGES,
    CHUNK_OVERLAP,
    PRE_EXTRACTION_WEBHOOK_URL,
    PRE_EXTRACTION_WEBHOOK_MIN_CHARS,
    PRE_EXTRACTION_WEBHOOK_TIMEOUT,
    logger,
)
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredEPubLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)


# Extensions that identify binary file formats handled by dedicated loaders.
# Used to prevent a conflicting multipart Content-Type (e.g. ``text/markdown``)
# from hijacking these files into a text loader.
_BINARY_FILE_EXTENSIONS = frozenset(
    {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "epub"}
)


def detect_file_encoding(filepath: str) -> str:
    """
    Detect the encoding of a file using BOM markers and chardet for broader support.
    Returns the detected encoding or 'utf-8' as default.
    """
    with open(filepath, "rb") as f:
        raw = f.read(4096)  # Read a larger sample for better detection

    # Check for BOM markers first
    if raw.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    elif raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    elif raw.startswith(codecs.BOM_UTF16):
        return "utf-16"
    elif raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    elif raw.startswith(codecs.BOM_UTF32_LE):
        return "utf-32-le"
    elif raw.startswith(codecs.BOM_UTF32_BE):
        return "utf-32-be"

    # Use chardet to detect encoding if no BOM is found
    result = chardet.detect(raw)
    encoding = result.get("encoding")
    if encoding:
        return encoding.lower()
    # Default to utf-8 if detection fails
    return "utf-8"


def cleanup_temp_encoding_file(loader) -> None:
    """
    Clean up temporary UTF-8 file if it was created for encoding conversion.

    :param loader: The document loader that may have created a temporary file
    """
    if hasattr(loader, "_temp_filepath") and loader._temp_filepath is not None:
        try:
            os.remove(loader._temp_filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary UTF-8 file: {e}")


def get_loader(
    filename: str,
    file_content_type: str,
    filepath: str,
    raw_text: bool = False,
):
    """Get the appropriate document loader based on file type and\or content type.

    When ``raw_text`` is True, text-formatted files (e.g. Markdown) are loaded
    verbatim with :class:`TextLoader` so their original formatting is
    preserved. This is intended for the ``/text`` endpoint, where the caller
    wants the raw file contents. The embedding path should keep the default
    (``raw_text=False``) so semantic loaders continue to strip formatting for
    better vector search quality.
    """
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    # File Content Type reference:
    # ref.: https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types/Common_types
    if file_ext == "pdf" or file_content_type == "application/pdf":
        loader = SafePyPDFLoader(filepath, extract_images=PDF_EXTRACT_IMAGES)
    elif file_ext == "csv" or file_content_type == "text/csv":
        # Detect encoding for CSV files
        encoding = detect_file_encoding(filepath)

        if encoding != "utf-8":
            # For non-UTF-8 encodings, convert to UTF-8 using streaming
            # to avoid holding the entire file in memory as a single string
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".csv", delete=False
                ) as temp_file:
                    with open(
                        filepath, "r", encoding=encoding, errors="replace"
                    ) as original_file:
                        while True:
                            chunk = original_file.read(64 * 1024)
                            if not chunk:
                                break
                            temp_file.write(chunk)

                    temp_filepath = temp_file.name

                loader = CSVLoader(temp_filepath)
                loader._temp_filepath = temp_filepath
            except Exception as e:
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise e
        else:
            loader = CSVLoader(filepath)
    elif file_ext == "rst":
        loader = UnstructuredRSTLoader(filepath, mode="elements")
    elif file_ext == "xml" or file_content_type in [
        "application/xml",
        "text/xml",
        "application/xhtml+xml",
    ]:
        loader = UnstructuredXMLLoader(filepath)
    elif file_ext in ["ppt", "pptx"] or file_content_type in [
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ]:
        loader = UnstructuredPowerPointLoader(filepath)
    elif file_ext == "md" or (
        file_content_type
        in [
            "text/markdown",
            "text/x-markdown",
            "application/markdown",
            "application/x-markdown",
        ]
        and file_ext not in _BINARY_FILE_EXTENSIONS
    ):
        if raw_text:
            loader = TextLoader(filepath, autodetect_encoding=True)
        else:
            loader = UnstructuredMarkdownLoader(filepath)
    elif file_ext == "epub" or file_content_type == "application/epub+zip":
        loader = UnstructuredEPubLoader(filepath)
    elif file_ext in ["doc", "docx"] or file_content_type in [
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        loader = Docx2txtLoader(filepath)
    elif file_ext in ["xls", "xlsx"] or file_content_type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]:
        loader = UnstructuredExcelLoader(filepath)
    elif file_ext == "json" or file_content_type == "application/json":
        loader = TextLoader(filepath, autodetect_encoding=True)
    elif file_ext in known_source_ext or (
        file_content_type and file_content_type.find("text/") >= 0
    ):
        loader = TextLoader(filepath, autodetect_encoding=True)
    else:
        loader = TextLoader(filepath, autodetect_encoding=True)
        known_type = False

    return loader, known_type, file_ext


def clean_text(text: str) -> str:
    """
    Clean up text from PDF lopader

    :param text: The original text
    :return: Cleaned text
    """
    text = remove_null(text)
    text = remove_non_utf8(text)
    return text


def remove_null(text: str) -> str:
    """
    Remove NUL (0x00) characters from a string.

    :param text: The original text with potential NUL characters.
    :return: Cleaned text without NUL characters.
    """
    return text.replace("\x00", "")


def remove_non_utf8(text: str) -> str:
    """
    Remove invalid UTF-8 characters from a string, such as surrogate characters

    :param text: The original text with potential invalid utf-8 characters
    :return: Cleaned text without invalid utf-8 characters.
    """
    try:
        return text.encode("utf-8", "ignore").decode("utf-8")
    except UnicodeError:
        return text


def maybe_enrich_with_webhook(
    file_path: str, documents: List[Document]
) -> List[Document]:
    """
    Optional hook: when PRE_EXTRACTION_WEBHOOK_URL is set and the current
    extraction returned pages averaging fewer than
    PRE_EXTRACTION_WEBHOOK_MIN_CHARS characters, POST the original file to the
    webhook and substitute its returned text. On any failure the original
    documents are returned unchanged (soft-fail by design).
    """
    url = PRE_EXTRACTION_WEBHOOK_URL
    if not url:
        return documents

    # Compute average characters per extracted page; empty list counts as 0.
    if documents:
        avg_chars = sum(
            len((doc.page_content or "").strip()) for doc in documents
        ) / len(documents)
    else:
        avg_chars = 0

    if avg_chars >= PRE_EXTRACTION_WEBHOOK_MIN_CHARS:
        return documents

    # Import lazily so module import works in environments without `requests`
    # (e.g. test collection phase when the feature is disabled).
    import requests

    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                url,
                files={"file": (os.path.basename(file_path), f)},
                timeout=PRE_EXTRACTION_WEBHOOK_TIMEOUT,
            )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # broad by intent — never break ingest
        logger.warning(
            "pre-extraction webhook failed, falling back to original text: %s", exc
        )
        return documents

    text = (payload.get("text") or "").strip()
    if not text:
        logger.warning(
            "pre-extraction webhook returned empty text, keeping original extraction"
        )
        return documents

    provider = payload.get("provider", "unknown")
    source = (
        documents[0].metadata.get("source")
        if documents and isinstance(documents[0].metadata, dict)
        else file_path
    )

    logger.info(
        "pre-extraction webhook enriched %s with %d chars from provider %s",
        file_path,
        len(text),
        provider,
    )

    return [
        Document(
            page_content=text,
            metadata={
                "source": source,
                "ocr_used": True,
                "ocr_provider": provider,
            },
        )
    ]


def process_documents(documents: List[Document]) -> str:
    processed_text = ""
    last_page: Optional[int] = None
    doc_basename = ""

    for doc in documents:
        if "source" in doc.metadata:
            doc_basename = doc.metadata["source"].split("/")[-1]
            break

    processed_text += f"{doc_basename}\n"

    for doc in documents:
        current_page = doc.metadata.get("page")
        if current_page and current_page != last_page:
            processed_text += f"\n# PAGE {doc.metadata['page']}\n\n"
            last_page = current_page

        new_content = doc.page_content
        if processed_text.endswith(new_content[:CHUNK_OVERLAP]):
            processed_text += new_content[CHUNK_OVERLAP:]
        else:
            processed_text += new_content

    return processed_text.strip()


class SafePyPDFLoader:
    """
    A wrapper around PyPDFLoader that handles image extraction failures gracefully.
    Falls back to text-only extraction when image extraction fails.

    This is a workaround for issues with PyPDFLoader that can occur when extracting images
    from PDFs, which can lead to KeyError exceptions if the PDF is malformed or has unsupported
    image formats. This class attempts to load the PDF with image extraction enabled, and if it
    fails due to a KeyError related to image filters, it falls back to loading the PDF
    without image extraction.
    ref.: https://github.com/langchain-ai/langchain/issues/26652
    """

    def __init__(self, filepath: str, extract_images: bool = False):
        self.filepath = filepath
        self.extract_images = extract_images
        self._temp_filepath = None  # For compatibility with cleanup function

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load PDF documents with automatic fallback on image extraction errors."""
        loader = PyPDFLoader(self.filepath, extract_images=self.extract_images)

        if not self.extract_images:
            # No image extraction: no fallback needed, stream directly
            yield from loader.lazy_load()
            return

        # extract_images=True: must collect eagerly so that a mid-stream
        # KeyError doesn't leave already-yielded pages duplicated by the
        # fallback (yield from + try/except would deliver partial + full).
        try:
            pages = list(loader.lazy_load())
        except KeyError as e:
            if "/Filter" in str(e):
                logger.warning(
                    f"PDF image extraction failed for {self.filepath}, falling back to text-only: {e}"
                )
                fallback_loader = PyPDFLoader(self.filepath, extract_images=False)
                pages = list(fallback_loader.lazy_load())
            else:
                # Re-raise if it's a different error
                raise
        yield from pages

    def load(self) -> List[Document]:
        """Load PDF documents with automatic fallback on image extraction errors."""
        return list(self.lazy_load())
