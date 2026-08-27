import pytest

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.search import SearchResult


class FakeClient:
    def __init__(self):
        self.received_messages = None

    def ask(self, messages):
        self.received_messages = messages
        return "Grounded answer"

    def stream(self, messages):
        self.received_messages = messages
        yield "Grounded answer"


class FakeRepository:
    def __init__(self):
        self.saved = []

    def save(self, conversation):
        self.saved.append(conversation.history().copy())

    def load(self):
        return Conversation()


class FakeChunker:
    def __init__(self, chunks):
        self.chunks = chunks
        self.documents = []

    def split(self, document):
        self.documents.append(document)
        return self.chunks


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.indexed = []
        self.deleted_documents = []
        self.queries = []

    def index_document(self, document_id, chunks):
        self.indexed.append((
            document_id,
            list(chunks),
        ))

    def delete_document(self, document_id):
        self.deleted_documents.append(document_id)
        return 2

    def retrieve(self, query, k=5):
        self.queries.append((query, k))

        if self.error:
            raise self.error

        return self.results


def make_chunk() -> Chunk:
    return Chunk(
        id="chunk_context",
        document_id="doc_rag",
        content="Retrieved knowledge",
        index=0,
    )


def build_manager(
    *,
    retriever=None,
    chunker=None,
    retrieval_k=2,
):
    conversation = Conversation()
    client = FakeClient()
    repository = FakeRepository()
    chunker = chunker or FakeChunker([make_chunk()])
    retriever = retriever or FakeRetriever([
        SearchResult(make_chunk(), 0.9)
    ])
    manager = RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=repository,
        chunker=chunker,
        retriever=retriever,
        retrieval_k=retrieval_k,
    )
    return (
        manager,
        conversation,
        client,
        repository,
        chunker,
        retriever,
    )


def test_index_document_chunks_and_indexes_content():
    chunk = make_chunk()
    chunker = FakeChunker([chunk])
    retriever = FakeRetriever()
    manager, _, _, _, _, _ = build_manager(
        chunker=chunker,
        retriever=retriever,
    )
    document = Document(
        id="doc_rag",
        content="Document content",
    )

    chunks = manager.index_document(document)

    assert chunks == [chunk]
    assert chunker.documents == [document]
    assert retriever.indexed == [(
        document.id,
        [chunk],
    )]


def test_delete_document_delegates_to_retriever():
    retriever = FakeRetriever()
    manager, _, _, _, _, _ = build_manager(
        retriever=retriever
    )

    deleted_count = manager.delete_document("doc_rag")

    assert deleted_count == 2
    assert retriever.deleted_documents == ["doc_rag"]


def test_send_message_retrieves_context_before_llm_call():
    manager, conversation, client, repository, _, retriever = (
        build_manager()
    )

    response = manager.send_message("User question")

    assert response == "Grounded answer"
    assert retriever.queries == [("User question", 2)]
    assert "Retrieved knowledge" in (
        client.received_messages[-1]["content"]
    )
    assert "User question" in (
        client.received_messages[-1]["content"]
    )
    assert [
        message.content
        for message in conversation.history()
    ] == ["User question", "Grounded answer"]
    assert len(repository.saved) == 1


def test_retrieval_failure_rolls_back_user_message():
    retriever = FakeRetriever(
        error=RuntimeError("retrieval failed")
    )
    manager, conversation, client, repository, _, _ = (
        build_manager(retriever=retriever)
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        manager.send_message("Question")

    assert conversation.is_empty()
    assert client.received_messages is None
    assert repository.saved == []


def test_rag_manager_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="greater than zero"):
        build_manager(retrieval_k=0)
