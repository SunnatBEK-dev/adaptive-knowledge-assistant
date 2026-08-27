from collections.abc import Iterable
from dataclasses import dataclass

from ai_sdk.retrieval.chunk import Chunk


@dataclass(frozen=True)
class IndexedDocument:
    """A catalog summary derived from indexed chunks."""

    document_id: str
    source: str
    chunk_count: int
    content_hash: str | None = None
    ingestion_root: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError(
                "Indexed document ID cannot be empty."
            )

        if not self.source.strip():
            raise ValueError(
                "Indexed document source cannot be empty."
            )

        if self.chunk_count <= 0:
            raise ValueError(
                "Indexed document chunk count must be "
                "greater than zero."
            )

        if (
            self.content_hash is not None
            and not self.content_hash.strip()
        ):
            raise ValueError(
                "Indexed document content hash cannot be empty."
            )

        if (
            self.ingestion_root is not None
            and not self.ingestion_root.strip()
        ):
            raise ValueError(
                "Indexed document ingestion root cannot be empty."
            )


def build_document_catalog(
    chunks: Iterable[Chunk],
) -> list[IndexedDocument]:
    sources: dict[str, str] = {}
    counts: dict[str, int] = {}
    content_hashes: dict[str, str | None] = {}
    ingestion_roots: dict[str, str | None] = {}

    for chunk in chunks:
        source = chunk.metadata.get("source")

        if not source or not source.strip():
            source = chunk.document_id

        sources.setdefault(
            chunk.document_id,
            source,
        )
        content_hashes.setdefault(
            chunk.document_id,
            chunk.metadata.get("content_hash"),
        )
        ingestion_roots.setdefault(
            chunk.document_id,
            chunk.metadata.get("ingestion_root"),
        )
        counts[chunk.document_id] = (
            counts.get(chunk.document_id, 0) + 1
        )

    return [
        IndexedDocument(
            document_id=document_id,
            source=sources[document_id],
            chunk_count=counts[document_id],
            content_hash=content_hashes[document_id],
            ingestion_root=ingestion_roots[document_id],
        )
        for document_id in sorted(counts)
    ]
