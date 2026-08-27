from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, sqrt

from ai_sdk.embeddings.base import EmbeddingVector
from ai_sdk.retrieval.chunk import Chunk


EmbeddedChunk = tuple[Chunk, EmbeddingVector]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Measure vector direction similarity without external math libraries."""
    if not left or not right:
        raise ValueError(
            "Embedding vectors cannot be empty."
        )

    if len(left) != len(right):
        raise ValueError(
            "Embedding vectors must have equal dimensions."
        )

    dot_product = fsum(
        left_value * right_value
        for left_value, right_value in zip(
            left,
            right,
            strict=True,
        )
    )
    left_norm = sqrt(fsum(
        value * value
        for value in left
    ))
    right_norm = sqrt(fsum(
        value * value
        for value in right
    ))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    score = dot_product / (
        left_norm * right_norm
    )

    return max(-1.0, min(1.0, score))


def top_k_search(
    query_vector: Sequence[float],
    candidates: Sequence[EmbeddedChunk],
    k: int = 5,
) -> list[SearchResult]:
    """Return the most similar chunks in descending score order."""
    if k <= 0:
        raise ValueError(
            "Top-k value must be greater than zero."
        )

    results = [
        SearchResult(
            chunk=chunk,
            score=cosine_similarity(
                query_vector,
                vector,
            ),
        )
        for chunk, vector in candidates
    ]
    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return results[:k]
