import json

import pytest

from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.json_store import JsonVectorStore


def make_chunk(
    chunk_id: str,
    content: str,
    index: int = 0,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc_json_store",
        content=content,
        index=index,
        metadata={"source": "qo‘llanma.txt"},
    )


def test_store_round_trip_preserves_chunks_and_searches(tmp_path):
    file_path = tmp_path / "nested" / "vectors.json"
    store = JsonVectorStore(file_path)
    python_chunk = make_chunk(
        "chunk_python",
        "Python functions",
    )
    cooking_chunk = make_chunk(
        "chunk_cooking",
        "Cooking recipes",
        1,
    )

    store.add_many([
        (python_chunk, [1.0, 0.0]),
        (cooking_chunk, [0.0, 1.0]),
    ])
    restarted = JsonVectorStore(file_path)
    results = restarted.search([1.0, 0.0], k=1)

    assert restarted.count() == 2
    assert results[0].chunk.id == "chunk_python"
    assert results[0].chunk.metadata == {
        "source": "qo‘llanma.txt"
    }
    assert results[0].score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="dimension"):
        restarted.search([1.0], k=1)

    payload = json.loads(file_path.read_text(
        encoding="utf-8"
    ))
    assert payload["version"] == 1


def test_store_replacement_delete_and_clear_survive_restart(
    tmp_path,
):
    file_path = tmp_path / "vectors.json"
    store = JsonVectorStore(file_path)
    store.add(
        make_chunk("chunk_same", "Original"),
        [1.0, 0.0],
    )
    store.add(
        make_chunk("chunk_same", "Replacement"),
        [0.0, 1.0],
    )

    restarted = JsonVectorStore(file_path)

    assert restarted.count() == 1
    assert restarted.search(
        [0.0, 1.0],
        k=1,
    )[0].chunk.content == "Replacement"
    assert restarted.delete("chunk_same") is True
    assert restarted.delete("chunk_same") is False

    empty = JsonVectorStore(file_path)
    empty.add(
        make_chunk("chunk_three", "Three dimensions"),
        [1.0, 0.0, 0.0],
    )
    empty.clear()

    assert JsonVectorStore(file_path).count() == 0


def test_add_many_rolls_back_on_invalid_dimension(tmp_path):
    file_path = tmp_path / "vectors.json"
    original = make_chunk("chunk_original", "Original")
    store = JsonVectorStore(file_path)
    store.add(original, [1.0, 0.0])

    with pytest.raises(ValueError, match="dimension"):
        store.add_many([
            (
                make_chunk("chunk_valid", "Valid", 1),
                [0.0, 1.0],
            ),
            (
                make_chunk("chunk_invalid", "Invalid", 2),
                [1.0, 0.0, 0.0],
            ),
        ])

    assert store.count() == 1
    assert JsonVectorStore(file_path).count() == 1


def test_add_rolls_back_when_file_write_fails(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "vectors.json"
    original = make_chunk("chunk_original", "Original")
    store = JsonVectorStore(file_path)
    store.add(original, [1.0, 0.0])

    def fail_to_save():
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "_save", fail_to_save)

    with pytest.raises(OSError, match="disk unavailable"):
        store.add(
            make_chunk("chunk_new", "New", 1),
            [0.0, 1.0],
        )

    assert store.count() == 1
    assert store.search(
        [1.0, 0.0],
        k=1,
    )[0].chunk.id == "chunk_original"
    assert JsonVectorStore(file_path).count() == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{invalid", "invalid JSON"),
        (json.dumps([]), "invalid format"),
        (
            json.dumps({"version": 99, "items": []}),
            "invalid format",
        ),
        (
            json.dumps({
                "version": 1,
                "items": [{
                    "chunk": {},
                    "vector": [1.0],
                }],
            }),
            "invalid format",
        ),
    ],
)
def test_store_rejects_corrupt_or_invalid_file(
    tmp_path,
    payload,
    message,
):
    file_path = tmp_path / "vectors.json"
    file_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        JsonVectorStore(file_path)


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([], "empty"),
        ([float("inf")], "finite"),
        (["not-a-number"], "numbers"),
    ],
)
def test_store_rejects_invalid_vector(
    tmp_path,
    vector,
    message,
):
    store = JsonVectorStore(tmp_path / "vectors.json")

    with pytest.raises(ValueError, match=message):
        store.add(
            make_chunk("chunk_invalid", "Invalid"),
            vector,
        )

    assert store.count() == 0
