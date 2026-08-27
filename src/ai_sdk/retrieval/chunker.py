from hashlib import sha256

from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.document import Document


class TextChunker:
    """Split document text with deterministic character windows."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(
                "Chunk size must be greater than zero."
            )

        if overlap < 0:
            raise ValueError(
                "Chunk overlap cannot be negative."
            )

        if overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than "
                "chunk size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        document: Document,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0

        while start < len(document.content):
            end = min(
                start + self.chunk_size,
                len(document.content),
            )
            content = document.content[start:end]

            if content.strip():
                chunks.append(Chunk(
                    id=self._create_chunk_id(
                        document_id=document.id,
                        start=start,
                        end=end,
                        content=content,
                    ),
                    document_id=document.id,
                    content=content,
                    index=len(chunks),
                    metadata=document.metadata,
                ))

            if end == len(document.content):
                break

            start += step

        return chunks

    @staticmethod
    def _create_chunk_id(
        document_id: str,
        start: int,
        end: int,
        content: str,
    ) -> str:
        identity = (
            f"{document_id}\0{start}\0{end}\0{content}"
        )
        digest = sha256(
            identity.encode("utf-8")
        ).hexdigest()[:12]

        return f"chunk_{digest}"
