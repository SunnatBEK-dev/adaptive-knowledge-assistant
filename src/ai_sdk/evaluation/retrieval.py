from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum
from typing import Protocol

from ai_sdk.retrieval.search import SearchResult


class _Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> Sequence[SearchResult]: ...


@dataclass(frozen=True)
class RetrievalEvalCase:
    """A query and the chunk IDs that count as relevant."""

    query: str
    expected_chunk_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError(
                "Evaluation query cannot be empty."
            )

        expected_chunk_ids = frozenset(
            self.expected_chunk_ids
        )

        if not expected_chunk_ids:
            raise ValueError(
                "Evaluation case must have at least one "
                "expected chunk ID."
            )

        if any(
            not chunk_id.strip()
            for chunk_id in expected_chunk_ids
        ):
            raise ValueError(
                "Expected chunk IDs cannot be empty."
            )

        object.__setattr__(
            self,
            "expected_chunk_ids",
            expected_chunk_ids,
        )


@dataclass(frozen=True)
class RetrievalEvalResult:
    """Per-query retrieval outcome at the configured top-k."""

    case: RetrievalEvalCase
    retrieved_chunk_ids: tuple[str, ...]
    hit: bool
    recall: float
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalEvalReport:
    """Aggregate retrieval metrics for one evaluation dataset."""

    k: int
    results: tuple[RetrievalEvalResult, ...]
    hit_rate: float
    mean_recall: float
    mean_reciprocal_rank: float

    @property
    def total_cases(self) -> int:
        return len(self.results)


class RetrievalEvaluator:
    """Measure Hit Rate, Recall, and MRR without an LLM judge."""

    def __init__(
        self,
        retriever: _Retriever,
        k: int = 5,
    ) -> None:
        if k <= 0:
            raise ValueError(
                "Evaluation top-k must be greater than zero."
            )

        self.retriever = retriever
        self.k = k

    def evaluate(
        self,
        cases: Sequence[RetrievalEvalCase],
    ) -> RetrievalEvalReport:
        case_list = tuple(cases)

        if not case_list:
            raise ValueError(
                "Evaluation dataset cannot be empty."
            )

        results = tuple(
            self._evaluate_case(case)
            for case in case_list
        )
        case_count = len(results)

        return RetrievalEvalReport(
            k=self.k,
            results=results,
            hit_rate=(
                fsum(result.hit for result in results)
                / case_count
            ),
            mean_recall=(
                fsum(
                    result.recall
                    for result in results
                )
                / case_count
            ),
            mean_reciprocal_rank=(
                fsum(
                    result.reciprocal_rank
                    for result in results
                )
                / case_count
            ),
        )

    def _evaluate_case(
        self,
        case: RetrievalEvalCase,
    ) -> RetrievalEvalResult:
        retrieved_chunk_ids = tuple(
            result.chunk.id
            for result in self.retriever.retrieve(
                case.query,
                k=self.k,
            )
        )
        relevant_ids = case.expected_chunk_ids.intersection(
            retrieved_chunk_ids
        )
        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            start=1,
        ):
            if chunk_id in case.expected_chunk_ids:
                reciprocal_rank = 1.0 / rank
                break

        return RetrievalEvalResult(
            case=case,
            retrieved_chunk_ids=retrieved_chunk_ids,
            hit=bool(relevant_ids),
            recall=(
                len(relevant_ids)
                / len(case.expected_chunk_ids)
            ),
            reciprocal_rank=reciprocal_rank,
        )
