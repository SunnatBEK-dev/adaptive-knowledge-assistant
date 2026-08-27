from dataclasses import dataclass

from ai_sdk.retrieval.search import SearchResult


@dataclass(frozen=True)
class Citation:
    """A local source reference for one retrieved chunk."""

    position: int
    document_id: str
    chunk_id: str
    source: str
    score: float

    def __post_init__(self) -> None:
        if self.position <= 0:
            raise ValueError(
                "Citation position must be greater than zero."
            )

    @classmethod
    def from_search_result(
        cls,
        position: int,
        result: SearchResult,
    ) -> "Citation":
        source = result.chunk.metadata.get("source")

        if not source or not source.strip():
            source = result.chunk.document_id

        return cls(
            position=position,
            document_id=result.chunk.document_id,
            chunk_id=result.chunk.id,
            source=source,
            score=result.score,
        )


@dataclass(frozen=True)
class RAGResponse:
    """Generated content paired with its retrieved citations."""

    content: str
    citations: tuple[Citation, ...]
