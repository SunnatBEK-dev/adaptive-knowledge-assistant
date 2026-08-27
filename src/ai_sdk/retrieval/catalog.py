from collections.abc import Iterable
from dataclasses import dataclass

from ai_sdk.retrieval.chunk import Chunk


@dataclass(frozen=True)
class IndexedDocument:
    """A catalog summary derived from indexed chunks."""

    document_id: str
    source: str
    chunk_count: int

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


def build_document_catalog(
    chunks: Iterable[Chunk],
) -> list[IndexedDocument]:
    sources: dict[str, str] = {}
    counts: dict[str, int] = {}

    for chunk in chunks:
        source = chunk.metadata.get("source")

        if not source or not source.strip():
            source = chunk.document_id

        sources.setdefault(
            chunk.document_id,
            source,
        )
        counts[chunk.document_id] = (
            counts.get(chunk.document_id, 0) + 1
        )

    return [
        IndexedDocument(
            document_id=document_id,
            source=sources[document_id],
            chunk_count=counts[document_id],
        )
        for document_id in sorted(counts)
    ]
