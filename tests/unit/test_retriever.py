import pytest

from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.in_memory import (
    InMemoryVectorStore,
)
from ai_sdk.retrieval.retriever import (
    SemanticRetriever,
)


class RecordingEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [
            self.vectors[text]
            for text in texts
        ]


class BrokenEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return []


def make_chunk(chunk_id, content, index):
    return Chunk(
        id=chunk_id,
        document_id="doc_retriever",
        content=content,
        index=index,
    )


def test_retriever_indexes_chunks_and_searches_query():
    embeddings = RecordingEmbeddingClient({
        "Python": [1.0, 0.0],
        "Cooking": [0.0, 1.0],
        "Python question": [1.0, 0.0],
    })
    store = InMemoryVectorStore()
    retriever = SemanticRetriever(embeddings, store)
    python_chunk = make_chunk(
        "chunk_python",
        "Python",
        0,
    )
    cooking_chunk = make_chunk(
        "chunk_cooking",
        "Cooking",
        1,
    )

    retriever.index([python_chunk, cooking_chunk])
    results = retriever.retrieve(
        "Python question",
        k=1,
    )

    assert store.count() == 2
    assert results[0].chunk is python_chunk
    assert embeddings.calls == [
        ["Python", "Cooking"],
        ["Python question"],
    ]


def test_retriever_skips_embedding_for_empty_batch():
    embeddings = RecordingEmbeddingClient()
    retriever = SemanticRetriever(
        embeddings,
        InMemoryVectorStore(),
    )

    retriever.index([])

    assert embeddings.calls == []


def test_retriever_rejects_embedding_cardinality_mismatch():
    store = InMemoryVectorStore()
    retriever = SemanticRetriever(
        BrokenEmbeddingClient(),
        store,
    )

    with pytest.raises(RuntimeError, match="each chunk"):
        retriever.index([
            make_chunk("chunk_one", "One", 0)
        ])

    assert store.count() == 0


def test_retriever_rejects_blank_query():
    retriever = SemanticRetriever(
        RecordingEmbeddingClient(),
        InMemoryVectorStore(),
    )

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("   ")
