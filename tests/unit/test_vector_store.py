import pytest

from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.in_memory import (
    InMemoryVectorStore,
)


def make_chunk(chunk_id: str, index: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc_store",
        content=f"Content {index}",
        index=index,
    )


def test_store_adds_copies_and_searches_vectors():
    store = InMemoryVectorStore()
    first = make_chunk("chunk_first", 0)
    second = make_chunk("chunk_second", 1)
    first_vector = [1.0, 0.0]

    store.add(first, first_vector)
    store.add(second, [0.0, 1.0])
    first_vector[0] = 0.0

    results = store.search([1.0, 0.0], k=1)

    assert store.count() == 2
    assert results[0].chunk is first
    assert results[0].score == pytest.approx(1.0)


def test_store_replaces_existing_chunk_without_growing():
    store = InMemoryVectorStore()
    original = make_chunk("chunk_same")
    replacement = Chunk(
        id="chunk_same",
        document_id="doc_store",
        content="Replacement",
        index=0,
    )

    store.add(original, [1.0, 0.0])
    store.add(replacement, [0.0, 1.0])

    assert store.count() == 1
    assert store.search(
        [0.0, 1.0],
        k=1,
    )[0].chunk is replacement


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([], "empty"),
        ([1.0, 2.0, 3.0], "dimension"),
    ],
)
def test_store_rejects_invalid_vector(
    vector,
    message,
):
    store = InMemoryVectorStore()
    store.add(make_chunk("chunk_first"), [1.0, 0.0])

    with pytest.raises(ValueError, match=message):
        store.add(make_chunk("chunk_invalid", 1), vector)

    assert store.count() == 1


def test_store_delete_and_clear_reset_dimension():
    store = InMemoryVectorStore()
    first = make_chunk("chunk_first")

    store.add(first, [1.0, 0.0])
    assert store.delete(first.id) is True
    assert store.delete(first.id) is False

    store.add(make_chunk("chunk_new"), [1.0, 2.0, 3.0])
    store.clear()

    assert store.count() == 0
    assert store.search([1.0], k=1) == []
