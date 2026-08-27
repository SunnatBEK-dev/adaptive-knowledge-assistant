from abc import ABC, abstractmethod
from collections.abc import Sequence

from ai_sdk.embeddings.base import EmbeddingVector
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import (
    EmbeddedChunk,
    SearchResult,
)


class BaseVectorStore(ABC):
    """Provider-neutral storage contract for embedded chunks."""

    @abstractmethod
    def add(
        self,
        chunk: Chunk,
        vector: EmbeddingVector,
    ) -> None:
        raise NotImplementedError

    def add_many(
        self,
        items: Sequence[EmbeddedChunk],
    ) -> None:
        for chunk, vector in items:
            self.add(chunk, vector)

    @abstractmethod
    def search(
        self,
        query_vector: EmbeddingVector,
        k: int = 5,
    ) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError
