from ai_sdk.embeddings.base import EmbeddingVector
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import (
    EmbeddedChunk,
    SearchResult,
    top_k_search,
)
from ai_sdk.retrieval.vector_store import (
    BaseVectorStore,
)


class InMemoryVectorStore(BaseVectorStore):
    """Small learning-stage vector store backed by process memory."""

    def __init__(self) -> None:
        self._items: dict[str, EmbeddedChunk] = {}
        self._dimension: int | None = None

    def add(
        self,
        chunk: Chunk,
        vector: EmbeddingVector,
    ) -> None:
        if not vector:
            raise ValueError(
                "Embedding vector cannot be empty."
            )

        stored_vector = [
            float(value)
            for value in vector
        ]
        dimension = len(stored_vector)

        if (
            self._dimension is not None
            and dimension != self._dimension
        ):
            raise ValueError(
                "Embedding vector dimension does not match "
                "the vector store."
            )

        if self._dimension is None:
            self._dimension = dimension

        self._items[chunk.id] = (
            chunk,
            stored_vector,
        )

    def search(
        self,
        query_vector: EmbeddingVector,
        k: int = 5,
    ) -> list[SearchResult]:
        if (
            self._dimension is not None
            and len(query_vector) != self._dimension
        ):
            raise ValueError(
                "Query vector dimension does not match "
                "the vector store."
            )

        return top_k_search(
            query_vector=query_vector,
            candidates=list(self._items.values()),
            k=k,
        )

    def delete(self, chunk_id: str) -> bool:
        if chunk_id not in self._items:
            return False

        del self._items[chunk_id]

        if not self._items:
            self._dimension = None

        return True

    def clear(self) -> None:
        self._items.clear()
        self._dimension = None

    def count(self) -> int:
        return len(self._items)
