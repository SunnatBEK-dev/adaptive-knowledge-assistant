import pytest

from ai_sdk.retrieval.catalog import (
    IndexedDocument,
    build_document_catalog,
)
from ai_sdk.retrieval.chunk import Chunk


def make_chunk(
    chunk_id,
    document_id,
    index,
    source=None,
):
    metadata = {}

    if source is not None:
        metadata["source"] = source

    return Chunk(
        id=chunk_id,
        document_id=document_id,
        content=f"Content {index}",
        index=index,
        metadata=metadata,
    )


def test_catalog_groups_chunks_and_preserves_source():
    catalog = build_document_catalog([
        make_chunk(
            "chunk_b",
            "doc_b",
            0,
            "/guides/b.md",
        ),
        make_chunk(
            "chunk_a_one",
            "doc_a",
            0,
            "/guides/a.txt",
        ),
        make_chunk(
            "chunk_a_two",
            "doc_a",
            1,
            "/guides/a.txt",
        ),
    ])

    assert catalog == [
        IndexedDocument(
            document_id="doc_a",
            source="/guides/a.txt",
            chunk_count=2,
        ),
        IndexedDocument(
            document_id="doc_b",
            source="/guides/b.md",
            chunk_count=1,
        ),
    ]


def test_catalog_uses_document_id_when_source_is_missing():
    catalog = build_document_catalog([
        make_chunk(
            "chunk_fallback",
            "doc_fallback",
            0,
        )
    ])

    assert catalog[0].source == "doc_fallback"


@pytest.mark.parametrize(
    ("document_id", "source", "chunk_count", "message"),
    [
        ("", "source", 1, "ID"),
        ("doc", "", 1, "source"),
        ("doc", "source", 0, "chunk count"),
    ],
)
def test_indexed_document_rejects_invalid_data(
    document_id,
    source,
    chunk_count,
    message,
):
    with pytest.raises(ValueError, match=message):
        IndexedDocument(
            document_id=document_id,
            source=source,
            chunk_count=chunk_count,
        )
