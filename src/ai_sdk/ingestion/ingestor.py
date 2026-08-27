from collections.abc import Sequence
from pathlib import Path

from ai_sdk.ingestion.base import BaseDocumentLoader
from ai_sdk.retrieval.document import Document


class DocumentIngestor:
    """Route files and directories to registered document loaders."""

    def __init__(
        self,
        loaders: Sequence[BaseDocumentLoader],
    ) -> None:
        self.loaders = tuple(loaders)

        if not self.loaders:
            raise ValueError(
                "Document ingestor requires at least one loader."
            )

    def ingest(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
    ) -> list[Document]:
        input_path = Path(path).expanduser()

        if not input_path.exists():
            raise FileNotFoundError(
                "Document path does not exist: "
                f"{input_path}"
            )

        if input_path.is_file():
            return [self._load_file(input_path)]

        if not input_path.is_dir():
            raise ValueError(
                "Document path must be a file or directory."
            )

        iterator = (
            input_path.rglob("*")
            if recursive
            else input_path.iterdir()
        )
        file_paths = sorted(
            (
                candidate
                for candidate in iterator
                if candidate.is_file()
                and self._find_loader(candidate) is not None
            ),
            key=lambda candidate: str(candidate),
        )

        if not file_paths:
            raise ValueError(
                "No supported document files were found."
            )

        return [
            self._load_file(file_path)
            for file_path in file_paths
        ]

    def _load_file(self, path: Path) -> Document:
        loader = self._find_loader(path)

        if loader is None:
            raise ValueError(
                "Unsupported document format: "
                f"{path.suffix or '<none>'}"
            )

        return loader.load(path)

    def _find_loader(
        self,
        path: Path,
    ) -> BaseDocumentLoader | None:
        for loader in self.loaders:
            if loader.supports(path):
                return loader

        return None
