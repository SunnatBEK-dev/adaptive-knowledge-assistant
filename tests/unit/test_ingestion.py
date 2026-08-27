from hashlib import sha256

import pytest

from ai_sdk.ingestion import (
    DocumentIngestor,
    TextDocumentLoader,
    create_default_ingestor,
)


def test_text_loader_supports_document_formats(tmp_path):
    file_path = tmp_path / "Guide.MD"
    file_path.write_text(
        "# Python guide",
        encoding="utf-8",
    )
    loader = TextDocumentLoader()

    document = loader.load(file_path)

    assert loader.supports(file_path) is True
    assert document.content == "# Python guide"
    assert document.metadata == {
        "source": str(file_path.resolve()),
        "format": "md",
        "content_hash": sha256(
            b"# Python guide"
        ).hexdigest(),
    }


def test_ingestor_loads_supported_directory_files_in_order(
    tmp_path,
):
    directory = tmp_path / "knowledge"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (directory / "b.md").write_text(
        "Markdown",
        encoding="utf-8",
    )
    (directory / "a.txt").write_text(
        "Text",
        encoding="utf-8",
    )
    (nested / "c.rst").write_text(
        "RST",
        encoding="utf-8",
    )
    (directory / "ignored.pdf").write_bytes(b"%PDF")

    documents = create_default_ingestor().ingest(
        directory
    )

    assert [
        document.metadata["format"]
        for document in documents
    ] == ["txt", "md", "rst"]
    assert [
        document.content
        for document in documents
    ] == ["Text", "Markdown", "RST"]
    assert {
        document.metadata["ingestion_root"]
        for document in documents
    } == {str(directory.resolve())}


def test_ingestor_can_disable_recursive_directory_scan(
    tmp_path,
):
    directory = tmp_path / "knowledge"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (directory / "top.txt").write_text(
        "Top",
        encoding="utf-8",
    )
    (nested / "nested.txt").write_text(
        "Nested",
        encoding="utf-8",
    )

    documents = create_default_ingestor().ingest(
        directory,
        recursive=False,
    )

    assert [document.content for document in documents] == [
        "Top"
    ]


def test_ingestor_rejects_unsupported_direct_file(tmp_path):
    file_path = tmp_path / "guide.pdf"
    file_path.write_bytes(b"%PDF")
    ingestor = DocumentIngestor([
        TextDocumentLoader()
    ])

    with pytest.raises(ValueError, match="Unsupported"):
        ingestor.ingest(file_path)


def test_ingestor_can_allow_empty_directory(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()

    documents = create_default_ingestor().ingest(
        directory,
        allow_empty=True,
    )

    assert documents == []
