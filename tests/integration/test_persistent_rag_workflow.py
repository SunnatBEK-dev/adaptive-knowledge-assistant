import pytest

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.json_store import JsonVectorStore
from ai_sdk.retrieval.retriever import (
    SemanticRetriever,
)
from ai_sdk.storage.json import JsonConversationRepository


pytestmark = pytest.mark.integration


class KeywordEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [
            [1.0, 0.0]
            if "python" in text.lower()
            else [0.0, 1.0]
            for text in texts
        ]


class RecordingLLMClient:
    def __init__(self):
        self.received_messages = None

    def ask(self, messages):
        self.received_messages = messages
        return "Grounded after restart"

    def stream(self, messages):
        raise NotImplementedError


def test_rag_reuses_persisted_index_after_restart(tmp_path):
    vector_path = tmp_path / "vectors.json"
    embedding_client = KeywordEmbeddingClient()
    first_retriever = SemanticRetriever(
        embedding_client=embedding_client,
        vector_store=JsonVectorStore(vector_path),
    )
    chunks = TextChunker(
        chunk_size=16,
        overlap=0,
    ).split(Document(
        id="doc_persistent_rag",
        content="Python functions Cooking recipes",
    ))
    first_retriever.index(chunks)

    restarted_retriever = SemanticRetriever(
        embedding_client=embedding_client,
        vector_store=JsonVectorStore(vector_path),
    )
    conversation = Conversation()
    client = RecordingLLMClient()
    manager = RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=JsonConversationRepository(
            tmp_path / "chat.json"
        ),
        chunker=TextChunker(),
        retriever=restarted_retriever,
        retrieval_k=1,
    )

    answer = manager.send_message(
        "How do Python functions work?"
    )

    assert answer == "Grounded after restart"
    assert len(chunks) == 2
    assert "Python functions" in (
        client.received_messages[-1]["content"]
    )
