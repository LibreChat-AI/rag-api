import os
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from app.utils.document_loader import (
    get_loader,
    clean_text,
    process_documents,
    detect_file_encoding,
)
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document


def test_clean_text():
    text = "Hello\x00World"
    cleaned = clean_text(text)
    assert "\x00" not in cleaned
    assert cleaned == "HelloWorld"


def test_get_loader_text(tmp_path):
    # Create a temporary text file.
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample text")
    loader, known_type, file_ext = get_loader("test.txt", "text/plain", str(file_path))
    assert known_type is True
    assert file_ext == "txt"
    data = loader.load()
    # Check that data is loaded.
    assert data is not None


def test_process_documents():
    docs = [
        Document(
            page_content="Page 1 content", metadata={"source": "dummy.txt", "page": 1}
        ),
        Document(
            page_content="Page 2 content", metadata={"source": "dummy.txt", "page": 2}
        ),
    ]
    processed = process_documents(docs)
    assert "dummy.txt" in processed
    assert "# PAGE 1" in processed
    assert "# PAGE 2" in processed


def test_safe_pdf_loader_class():
    """Test that SafePyPDFLoader class can be instantiated"""
    from app.utils.document_loader import SafePyPDFLoader

    # Test instantiation
    loader = SafePyPDFLoader("dummy.pdf", extract_images=True)
    assert loader.filepath == "dummy.pdf"
    assert loader.extract_images == True
    assert loader._temp_filepath is None


def test_get_loader_text_lazy_load(tmp_path):
    """Test that lazy_load returns an iterator yielding documents."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample text")
    loader, known_type, file_ext = get_loader("test.txt", "text/plain", str(file_path))
    assert known_type is True
    assert file_ext == "txt"
    data = list(loader.lazy_load())
    assert len(data) > 0
    assert hasattr(data[0], "page_content")


def test_get_loader_pdf(tmp_path):
    """Test get_loader returns SafePyPDFLoader for PDF files"""
    # Create a dummy PDF file path (doesn't need to be real for this test)
    file_path = tmp_path / "test.pdf"
    file_path.write_text("dummy content")  # Not a real PDF, but that's OK for this test

    loader, known_type, file_ext = get_loader(
        "test.pdf", "application/pdf", str(file_path)
    )

    # Check that we get our SafePyPDFLoader
    from app.utils.document_loader import SafePyPDFLoader

    assert isinstance(loader, SafePyPDFLoader)
    assert known_type is True
    assert file_ext == "pdf"


def test_safe_pdf_loader_lazy_load():
    """Test that SafePyPDFLoader.lazy_load() returns an Iterator."""
    from app.utils.document_loader import SafePyPDFLoader

    loader = SafePyPDFLoader("dummy.pdf", extract_images=False)
    assert hasattr(loader, "lazy_load")
    result = loader.lazy_load()
    assert isinstance(result, Iterator)


def test_safe_pdf_loader_fallback_no_duplicate_pages():
    """Fallback after mid-stream KeyError must not duplicate already-yielded pages."""
    from app.utils.document_loader import SafePyPDFLoader

    fallback_docs = [Document(page_content=f"fallback page {i}") for i in range(5)]

    def primary_gen():
        yield Document(page_content="partial page 0")
        yield Document(page_content="partial page 1")
        raise KeyError("/Filter")

    def fallback_gen():
        yield from fallback_docs

    loader = SafePyPDFLoader("dummy.pdf", extract_images=True)

    with patch("app.utils.document_loader.PyPDFLoader") as MockPDF:
        primary_instance = MagicMock()
        primary_instance.lazy_load.side_effect = primary_gen
        fallback_instance = MagicMock()
        fallback_instance.lazy_load.side_effect = fallback_gen
        MockPDF.side_effect = [primary_instance, fallback_instance]

        result = list(loader.lazy_load())

    # Must be exactly the 5 fallback pages, NOT 2 partial + 5 fallback = 7
    assert len(result) == 5
    assert result[0].page_content == "fallback page 0"
    assert result[-1].page_content == "fallback page 4"


def test_safe_pdf_loader_fallback_via_load():
    """load() delegates to lazy_load(), so fallback must also be correct via load()."""
    from app.utils.document_loader import SafePyPDFLoader

    fallback_docs = [Document(page_content=f"fb {i}") for i in range(3)]

    def primary_gen():
        yield Document(page_content="partial 0")
        raise KeyError("/Filter")

    def fallback_gen():
        yield from fallback_docs

    loader = SafePyPDFLoader("dummy.pdf", extract_images=True)

    with patch("app.utils.document_loader.PyPDFLoader") as MockPDF:
        primary_instance = MagicMock()
        primary_instance.lazy_load.side_effect = primary_gen
        fallback_instance = MagicMock()
        fallback_instance.lazy_load.side_effect = fallback_gen
        MockPDF.side_effect = [primary_instance, fallback_instance]

        result = loader.load()

    assert len(result) == 3
    assert result[0].page_content == "fb 0"


def test_safe_pdf_loader_non_filter_error_propagates():
    """KeyError that isn't /Filter should propagate, not silently fallback."""
    from app.utils.document_loader import SafePyPDFLoader

    def bad_gen():
        raise KeyError("SomeOtherKey")

    loader = SafePyPDFLoader("dummy.pdf", extract_images=True)

    with patch("app.utils.document_loader.PyPDFLoader") as MockPDF:
        instance = MagicMock()
        instance.lazy_load.side_effect = bad_gen
        MockPDF.return_value = instance

        with pytest.raises(KeyError, match="SomeOtherKey"):
            list(loader.lazy_load())


MARKDOWN_SAMPLE = (
    "# Heading\n\n"
    "**bold** and *italic* text with a [link](https://example.com).\n\n"
    "- item 1\n"
    "- item 2\n\n"
    "> a blockquote\n"
)


def test_get_loader_markdown_embed_uses_unstructured(tmp_path):
    """Default (embedding) path must keep UnstructuredMarkdownLoader for .md."""
    file_path = tmp_path / "notes.md"
    file_path.write_text(MARKDOWN_SAMPLE, encoding="utf-8")

    loader, known_type, file_ext = get_loader(
        "notes.md", "text/markdown", str(file_path)
    )

    assert isinstance(loader, UnstructuredMarkdownLoader)
    assert known_type is True
    assert file_ext == "md"


@pytest.mark.parametrize(
    "content_type",
    [
        "text/markdown",
        "text/x-markdown",
        "application/markdown",
        "application/x-markdown",
    ],
)
def test_get_loader_markdown_raw_text_uses_text_loader(tmp_path, content_type):
    """/text path (raw_text=True) must load .md verbatim so formatting survives."""
    file_path = tmp_path / "notes.md"
    file_path.write_text(MARKDOWN_SAMPLE, encoding="utf-8")

    loader, known_type, file_ext = get_loader(
        "notes.md", content_type, str(file_path), raw_text=True
    )

    assert isinstance(loader, TextLoader)
    assert known_type is True
    assert file_ext == "md"

    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].page_content == MARKDOWN_SAMPLE


def test_get_loader_markdown_raw_text_by_extension_only(tmp_path):
    """Extension-based detection must still kick in when content type is generic."""
    file_path = tmp_path / "README.md"
    file_path.write_text(MARKDOWN_SAMPLE, encoding="utf-8")

    loader, _, _ = get_loader(
        "README.md", "application/octet-stream", str(file_path), raw_text=True
    )

    assert isinstance(loader, TextLoader)


def test_get_loader_raw_text_leaves_pdf_alone(tmp_path):
    """raw_text must not disturb binary formats — PDF still uses the PDF loader."""
    from app.utils.document_loader import SafePyPDFLoader

    file_path = tmp_path / "doc.pdf"
    file_path.write_text("not a real pdf")

    loader, _, file_ext = get_loader(
        "doc.pdf", "application/pdf", str(file_path), raw_text=True
    )

    assert isinstance(loader, SafePyPDFLoader)
    assert file_ext == "pdf"


@pytest.mark.parametrize(
    "filename, expected_loader_name",
    [
        ("doc.pdf", "SafePyPDFLoader"),
        ("report.docx", "Docx2txtLoader"),
        ("book.epub", "UnstructuredEPubLoader"),
        ("data.xlsx", "UnstructuredExcelLoader"),
        ("slides.pptx", "UnstructuredPowerPointLoader"),
    ],
)
def test_get_loader_raw_text_respects_binary_extensions_over_markdown_mime(
    tmp_path, filename, expected_loader_name
):
    """A markdown Content-Type must not override a known binary extension.

    Some clients send conflicting multipart content types. For an upload named
    `doc.pdf` with Content-Type `text/markdown`, the PDF loader still has to
    win — otherwise a binary file is read as UTF-8 text.
    """
    file_path = tmp_path / filename
    file_path.write_text("placeholder binary content")

    loader, _, _ = get_loader(
        filename, "text/markdown", str(file_path), raw_text=True
    )

    assert type(loader).__name__ == expected_loader_name


# ==================== Encoding Detection Tests ====================


def test_detect_file_encoding_utf8(tmp_path):
    """UTF-8 file should be detected correctly."""
    file_path = tmp_path / "test.csv"
    file_path.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
    assert detect_file_encoding(str(file_path)) == "utf-8"


def test_detect_file_encoding_utf8_bom(tmp_path):
    """UTF-8 with BOM should be detected as utf-8-sig."""
    file_path = tmp_path / "test.csv"
    file_path.write_bytes(b"\xef\xbb\xbfname,age\nAlice,30")
    assert detect_file_encoding(str(file_path)) == "utf-8-sig"


def test_detect_file_encoding_utf16_le(tmp_path):
    """UTF-16 LE with BOM should be detected correctly."""
    file_path = tmp_path / "test.csv"
    content = "name,age\nAlice,30".encode("utf-16-le")
    file_path.write_bytes(b"\xff\xfe" + content)
    assert detect_file_encoding(str(file_path)) == "utf-16-le"


def test_detect_file_encoding_shift_jis(tmp_path):
    """Shift-JIS file with ASCII headers and Japanese data should be detected."""
    # Simulate a CSV with English headers and Japanese data rows
    # This is the exact scenario from issue #291
    header = b"name,city\n"
    # Japanese text in Shift-JIS: "Tokyo" = "東京" = 8b 93 8b 5f
    japanese_data = b"Alice,\x8b\x93\x8b\x5f\nBob,Osaka\n"
    file_path = tmp_path / "test.csv"
    file_path.write_bytes(header + japanese_data * 100)  # Repeat for enough sample

    encoding = detect_file_encoding(str(file_path))
    # Should detect as cp932/shift_jis, NOT utf-8
    assert encoding in ["cp932", "shift_jis", "shift_jisx0213"], (
        f"Expected Shift-JIS variant, got: {encoding}"
    )


def test_detect_file_encoding_gb18030(tmp_path):
    """GB18030 file with ASCII headers and Chinese data should be detected."""
    header = b"name,city\n"
    # Chinese text in GB18030: "Beijing" = "北京" = b1 b1 be a9
    chinese_data = b"Alice,\xb1\xb1\xbe\xa9\nBob,Shanghai\n"
    file_path = tmp_path / "test.csv"
    file_path.write_bytes(header + chinese_data * 100)

    encoding = detect_file_encoding(str(file_path))
    # Should detect as gb18030 or gbk, NOT utf-8
    assert encoding in ["gb18030", "gbk", "gb2312"], (
        f"Expected GB variant, got: {encoding}"
    )


def test_detect_file_encoding_latin1_fallback(tmp_path):
    """File with Latin-1 characters should fallback correctly."""
    header = b"name,city\n"
    # Latin-1 specific bytes (0x80-0xFF range that's invalid UTF-8)
    latin1_data = b"Alice,Caf\xe9\nBob,Z\xfc\n"
    file_path = tmp_path / "test.csv"
    file_path.write_bytes(header + latin1_data * 100)

    encoding = detect_file_encoding(str(file_path))
    # Should detect as latin-1 or iso-8859-1
    assert encoding in ["latin-1", "iso-8859-1", "windows-1252", "ascii"], (
        f"Expected Latin variant, got: {encoding}"
    )


def test_get_loader_csv_shift_jis(tmp_path):
    """CSVLoader should handle Shift-JIS files without UnicodeDecodeError."""
    header = b"name,city\n"
    # Japanese text in Shift-JIS
    japanese_data = b"Alice,\x8b\x93\x8b\x5f\nBob,Osaka\n"
    file_path = tmp_path / "test.csv"
    file_path.write_bytes(header + japanese_data * 100)

    # This should NOT raise UnicodeDecodeError
    loader, known_type, file_ext = get_loader("test.csv", "text/csv", str(file_path))
    assert known_type is True
    assert file_ext == "csv"

    # Should be able to load without error
    data = loader.load()
    assert len(data) > 0
