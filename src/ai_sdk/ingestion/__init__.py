from ai_sdk.ingestion.base import BaseDocumentLoader
from ai_sdk.ingestion.ingestor import DocumentIngestor
from ai_sdk.ingestion.text import TextDocumentLoader


def create_default_ingestor() -> DocumentIngestor:
    return DocumentIngestor([
        TextDocumentLoader()
    ])


__all__ = [
    "BaseDocumentLoader",
    "DocumentIngestor",
    "TextDocumentLoader",
    "create_default_ingestor",
]
