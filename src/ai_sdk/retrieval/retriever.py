from collections.abc import Sequence

from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import SearchResult
from ai_sdk.retrieval.vector_store import (
    BaseVectorStore,
)


class SemanticRetriever:
    """Coordinate embedding generation and vector-store search."""

    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        vector_store: BaseVectorStore,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def index(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        chunk_list = list(chunks)

        if not chunk_list:
            return

        vectors = self.embedding_client.embed([
            chunk.content
            for chunk in chunk_list
        ])

        if len(vectors) != len(chunk_list):
            raise RuntimeError(
                "Embedding client must return one vector "
                "for each chunk."
            )

        self.vector_store.add_many(list(zip(
            chunk_list,
            vectors,
            strict=True,
        )))

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError(
                "Retrieval query cannot be empty."
            )

        query_vector = (
            self.embedding_client.embed_one(query)
        )

        return self.vector_store.search(
            query_vector=query_vector,
            k=k,
        )
