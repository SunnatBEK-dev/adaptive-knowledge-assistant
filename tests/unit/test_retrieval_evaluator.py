import pytest

from ai_sdk.evaluation.retrieval import (
    RetrievalEvalCase,
    RetrievalEvaluator,
)
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import SearchResult


class StubRetriever:
    def __init__(self, rankings):
        self.rankings = rankings
        self.calls = []

    def retrieve(self, query, k=5):
        self.calls.append((query, k))
        return self.rankings.get(query, [])[:k]


def make_result(chunk_id, index):
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            document_id="doc_eval",
            content=f"Content {index}",
            index=index,
        ),
        score=1.0 - (index / 10),
    )


def test_evaluator_calculates_per_case_and_aggregate_metrics():
    retriever = StubRetriever({
        "partial": [
            make_result("chunk_other", 0),
            make_result("chunk_relevant", 1),
        ],
        "exact": [make_result("chunk_exact", 0)],
        "miss": [],
    })
    cases = [
        RetrievalEvalCase(
            query="partial",
            expected_chunk_ids=frozenset({
                "chunk_relevant",
                "chunk_missing",
            }),
        ),
        RetrievalEvalCase(
            query="exact",
            expected_chunk_ids=frozenset({
                "chunk_exact"
            }),
        ),
        RetrievalEvalCase(
            query="miss",
            expected_chunk_ids=frozenset({
                "chunk_absent"
            }),
        ),
    ]

    report = RetrievalEvaluator(
        retriever,
        k=2,
    ).evaluate(cases)

    assert report.k == 2
    assert report.total_cases == 3
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mean_recall == pytest.approx(0.5)
    assert report.mean_reciprocal_rank == pytest.approx(0.5)
    assert report.results[0].retrieved_chunk_ids == (
        "chunk_other",
        "chunk_relevant",
    )
    assert report.results[0].hit is True
    assert report.results[0].recall == pytest.approx(0.5)
    assert report.results[0].reciprocal_rank == pytest.approx(
        0.5
    )
    assert report.results[2].hit is False
    assert retriever.calls == [
        ("partial", 2),
        ("exact", 2),
        ("miss", 2),
    ]


@pytest.mark.parametrize(
    ("query", "expected_ids", "message"),
    [
        ("   ", {"chunk_one"}, "query"),
        ("question", set(), "at least one"),
        ("question", {""}, "IDs"),
    ],
)
def test_eval_case_rejects_invalid_data(
    query,
    expected_ids,
    message,
):
    with pytest.raises(ValueError, match=message):
        RetrievalEvalCase(
            query=query,
            expected_chunk_ids=frozenset(expected_ids),
        )


def test_evaluator_rejects_non_positive_k():
    with pytest.raises(ValueError, match="greater than zero"):
        RetrievalEvaluator(StubRetriever({}), k=0)


def test_evaluator_rejects_empty_dataset():
    evaluator = RetrievalEvaluator(StubRetriever({}))

    with pytest.raises(ValueError, match="dataset"):
        evaluator.evaluate([])
