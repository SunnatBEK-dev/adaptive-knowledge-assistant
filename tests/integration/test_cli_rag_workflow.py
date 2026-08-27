import pytest

from app.main import load_document, run_cli
from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.json_store import JsonVectorStore
from ai_sdk.retrieval.retriever import SemanticRetriever
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
        raise NotImplementedError

    def stream(self, messages):
        self.received_messages = messages
        yield "Grounded CLI answer"


def test_cli_indexes_uses_lists_and_removes_document(
    tmp_path,
    capsys,
):
    guide_path = tmp_path / "Python guide.txt"
    guide_path.write_text(
        "Python functions contain reusable logic.",
        encoding="utf-8",
    )
    document_id = load_document(str(guide_path)).id
    vector_store = JsonVectorStore(
        tmp_path / "vectors.json"
    )
    retriever = SemanticRetriever(
        embedding_client=KeywordEmbeddingClient(),
        vector_store=vector_store,
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
        chunker=TextChunker(
            chunk_size=100,
            overlap=0,
        ),
        retriever=retriever,
        retrieval_k=1,
    )
    commands = iter([
        f"/index {guide_path}",
        "/documents",
        "How do Python functions work?",
        f"/remove {document_id}",
        "/exit",
    ])

    run_cli(
        manager,
        input_fn=lambda _: next(commands),
    )
    output = capsys.readouterr().out

    assert f"Indexed {document_id}: 1 chunks." in output
    assert f"- {document_id}" in output
    assert "Grounded CLI answer" in output
    assert "Sources:" in output
    assert f"[1] {guide_path.resolve()}" in output
    assert f"Removed {document_id}: 1 chunks." in output
    assert "Python functions contain reusable logic." in (
        client.received_messages[-1]["content"]
    )
    assert str(guide_path.resolve()) not in (
        client.received_messages[-1]["content"]
    )
    assert "Cite supporting context with [n]" in (
        client.received_messages[-1]["content"]
    )
    assert vector_store.count() == 0
