import pytest

from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.evaluation.retrieval import (
    RetrievalEvalCase,
    RetrievalEvaluator,
)
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.in_memory import (
    InMemoryVectorStore,
)
from ai_sdk.retrieval.retriever import (
    SemanticRetriever,
)


pytestmark = pytest.mark.integration


class KeywordEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [
            [1.0, 0.0]
            if "python" in text.lower()
            else [0.0, 1.0]
            for text in texts
        ]


def test_real_retriever_scores_offline_evaluation_dataset():
    retriever = SemanticRetriever(
        embedding_client=KeywordEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
    )
    retriever.index([
        Chunk(
            id="chunk_python",
            document_id="doc_guide",
            content="Python functions",
            index=0,
        ),
        Chunk(
            id="chunk_cooking",
            document_id="doc_guide",
            content="Cooking recipes",
            index=1,
        ),
    ])
    cases = [
        RetrievalEvalCase(
            query="How do Python functions work?",
            expected_chunk_ids=frozenset({
                "chunk_python"
            }),
        ),
        RetrievalEvalCase(
            query="How should I start cooking?",
            expected_chunk_ids=frozenset({
                "chunk_cooking"
            }),
        ),
    ]

    report = RetrievalEvaluator(
        retriever=retriever,
        k=1,
    ).evaluate(cases)

    assert report.total_cases == 2
    assert report.hit_rate == pytest.approx(1.0)
    assert report.mean_recall == pytest.approx(1.0)
    assert report.mean_reciprocal_rank == pytest.approx(1.0)
