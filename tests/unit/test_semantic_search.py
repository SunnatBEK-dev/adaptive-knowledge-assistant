import pytest

from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import (
    cosine_similarity,
    top_k_search,
)


def make_chunk(chunk_id: str, index: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc_search",
        content=f"Content {index}",
        index=index,
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
    ],
)
def test_cosine_similarity_known_directions(
    left,
    right,
    expected,
):
    assert cosine_similarity(
        left,
        right,
    ) == pytest.approx(expected)


def test_cosine_similarity_treats_zero_vector_as_no_similarity():
    assert cosine_similarity(
        [0.0, 0.0],
        [1.0, 0.0],
    ) == 0.0


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([], [], "empty"),
        ([1.0], [1.0, 2.0], "dimensions"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    left,
    right,
    message,
):
    with pytest.raises(ValueError, match=message):
        cosine_similarity(left, right)


def test_top_k_search_returns_highest_scoring_chunks():
    first = make_chunk("chunk_first", 0)
    second = make_chunk("chunk_second", 1)
    third = make_chunk("chunk_third", 2)

    results = top_k_search(
        query_vector=[1.0, 0.0],
        candidates=[
            (third, [0.0, 1.0]),
            (second, [0.8, 0.2]),
            (first, [1.0, 0.0]),
        ],
        k=2,
    )

    assert [result.chunk for result in results] == [
        first,
        second,
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score > 0.0


def test_top_k_search_preserves_input_order_for_equal_scores():
    first = make_chunk("chunk_first", 0)
    second = make_chunk("chunk_second", 1)

    results = top_k_search(
        query_vector=[1.0, 0.0],
        candidates=[
            (first, [1.0, 0.0]),
            (second, [1.0, 0.0]),
        ],
        k=2,
    )

    assert [result.chunk for result in results] == [
        first,
        second,
    ]


def test_top_k_search_returns_empty_for_no_candidates():
    assert top_k_search(
        query_vector=[1.0],
        candidates=[],
        k=3,
    ) == []


def test_top_k_search_rejects_non_positive_k():
    with pytest.raises(ValueError, match="greater than zero"):
        top_k_search(
            query_vector=[1.0],
            candidates=[],
            k=0,
        )
