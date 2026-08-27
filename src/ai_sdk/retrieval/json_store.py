import json
from collections.abc import Sequence
from json import JSONDecodeError
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

from ai_sdk.embeddings.base import EmbeddingVector
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.search import (
    EmbeddedChunk,
    SearchResult,
    top_k_search,
)
from ai_sdk.retrieval.vector_store import (
    BaseVectorStore,
)


class JsonVectorStore(BaseVectorStore):
    """Persist embedded chunks in a versioned local JSON file."""

    FORMAT_VERSION = 1

    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)
        self._items: dict[str, EmbeddedChunk] = {}
        self._dimension: int | None = None
        self._load()

    def add(
        self,
        chunk: Chunk,
        vector: EmbeddingVector,
    ) -> None:
        previous_items = self._items.copy()
        previous_dimension = self._dimension

        try:
            self._add_in_memory(chunk, vector)
            self._save()
        except Exception:
            self._items = previous_items
            self._dimension = previous_dimension
            raise

    def add_many(
        self,
        items: Sequence[EmbeddedChunk],
    ) -> None:
        item_list = list(items)

        if not item_list:
            return

        previous_items = self._items.copy()
        previous_dimension = self._dimension

        try:
            for chunk, vector in item_list:
                self._add_in_memory(chunk, vector)

            self._save()
        except Exception:
            self._items = previous_items
            self._dimension = previous_dimension
            raise

    def search(
        self,
        query_vector: EmbeddingVector,
        k: int = 5,
    ) -> list[SearchResult]:
        if (
            self._dimension is not None
            and len(query_vector) != self._dimension
        ):
            raise ValueError(
                "Query vector dimension does not match "
                "the vector store."
            )

        return top_k_search(
            query_vector=query_vector,
            candidates=list(self._items.values()),
            k=k,
        )

    def delete(self, chunk_id: str) -> bool:
        if chunk_id not in self._items:
            return False

        previous_items = self._items.copy()
        previous_dimension = self._dimension
        del self._items[chunk_id]

        if not self._items:
            self._dimension = None

        try:
            self._save()
        except Exception:
            self._items = previous_items
            self._dimension = previous_dimension
            raise

        return True

    def clear(self) -> None:
        previous_items = self._items.copy()
        previous_dimension = self._dimension
        self._items.clear()
        self._dimension = None

        try:
            self._save()
        except Exception:
            self._items = previous_items
            self._dimension = previous_dimension
            raise

    def count(self) -> int:
        return len(self._items)

    def _add_in_memory(
        self,
        chunk: Chunk,
        vector: EmbeddingVector,
    ) -> None:
        stored_vector = self._validate_vector(vector)
        dimension = len(stored_vector)

        if (
            self._dimension is not None
            and dimension != self._dimension
        ):
            raise ValueError(
                "Embedding vector dimension does not match "
                "the vector store."
            )

        if self._dimension is None:
            self._dimension = dimension

        self._items[chunk.id] = (
            chunk,
            stored_vector,
        )

    @staticmethod
    def _validate_vector(
        vector: EmbeddingVector,
    ) -> EmbeddingVector:
        if not vector:
            raise ValueError(
                "Embedding vector cannot be empty."
            )

        try:
            stored_vector = [
                float(value)
                for value in vector
            ]
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Embedding vector must contain numbers."
            ) from error

        if not all(
            isfinite(value)
            for value in stored_vector
        ):
            raise ValueError(
                "Embedding vector values must be finite."
            )

        return stored_vector

    def _load(self) -> None:
        if not self.file_path.exists():
            return

        try:
            with self.file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)
        except (JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                "Vector store file contains invalid JSON."
            ) from error

        try:
            self._restore(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Vector store file has an invalid format."
            ) from error

    def _restore(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise ValueError(
                "Vector store payload must be an object."
            )

        if payload.get("version") != self.FORMAT_VERSION:
            raise ValueError(
                "Vector store version is not supported."
            )

        records = payload.get("items")

        if not isinstance(records, list):
            raise ValueError(
                "Vector store items must be a list."
            )

        for record in records:
            chunk, vector = self._parse_record(record)

            if chunk.id in self._items:
                raise ValueError(
                    "Vector store chunk IDs must be unique."
                )

            self._add_in_memory(chunk, vector)

    @staticmethod
    def _parse_record(
        record: object,
    ) -> EmbeddedChunk:
        if not isinstance(record, dict):
            raise ValueError(
                "Vector store item must be an object."
            )

        chunk_data = record.get("chunk")
        vector = record.get("vector")

        if not isinstance(chunk_data, dict):
            raise ValueError(
                "Vector store chunk must be an object."
            )

        if not isinstance(vector, list):
            raise ValueError(
                "Vector store vector must be a list."
            )

        chunk = Chunk(
            id=chunk_data["id"],
            document_id=chunk_data["document_id"],
            content=chunk_data["content"],
            index=chunk_data["index"],
            metadata=chunk_data.get("metadata", {}),
        )

        return chunk, vector

    def _save(self) -> None:
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        payload = {
            "version": self.FORMAT_VERSION,
            "items": [
                {
                    "chunk": {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "index": chunk.index,
                        "metadata": chunk.metadata,
                    },
                    "vector": vector,
                }
                for chunk, vector in self._items.values()
            ],
        }
        temporary_path: Path | None = None

        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.file_path.parent,
                prefix=f".{self.file_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                json.dump(
                    payload,
                    file,
                    indent=4,
                    ensure_ascii=False,
                    allow_nan=False,
                )

            temporary_path.replace(self.file_path)
        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()
