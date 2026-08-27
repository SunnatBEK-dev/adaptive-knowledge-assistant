from hashlib import sha256

import pytest

from app.main import load_document


def test_load_document_uses_stable_path_identity(tmp_path):
    file_path = tmp_path / "Python guide.txt"
    file_path.write_text(
        "Python functions",
        encoding="utf-8",
    )

    first = load_document(str(file_path))
    file_path.write_text(
        "Updated Python functions",
        encoding="utf-8",
    )
    second = load_document(str(file_path))

    assert first.id == second.id
    assert first.id.startswith("doc_")
    assert first.content == "Python functions"
    assert second.content == "Updated Python functions"
    assert second.metadata == {
        "source": str(file_path.resolve()),
        "format": "txt",
        "content_hash": sha256(
            b"Updated Python functions"
        ).hexdigest(),
    }


def test_load_document_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_document(str(tmp_path / "missing.txt"))


def test_load_document_rejects_non_utf8_file(tmp_path):
    file_path = tmp_path / "binary.txt"
    file_path.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="UTF-8"):
        load_document(str(file_path))
