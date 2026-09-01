from threading import Event

import pytest

import ai_sdk.web.service as service_module
from ai_sdk.agents import (
    WorkflowProgressEvent,
    WorkflowProgressStatus,
)
from ai_sdk.application.bootstrap import AssistantRuntimeResources
from ai_sdk.application.rag_response import Citation, RAGResponse
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.ingestion import create_default_ingestor
from ai_sdk.memory.json_store import JSONMemoryStore
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.hybrid import HybridRetriever
from ai_sdk.retrieval.in_memory import InMemoryVectorStore
from ai_sdk.web.models import ChatStreamRequest, ConversationResetRequest
from ai_sdk.web.service import (
    ActiveRunError,
    KnowledgeAssistantService,
    UploadValidationError,
)


class DeterministicEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [
            [
                float(len(text)),
                float(sum(ord(char) for char in text) % 997),
            ]
            for text in texts
        ]


class FakeManager:
    def __init__(self, progress=None, error=None):
        self.client = object()
        self.progress = progress
        self.error = error

    def send_message_with_citations(self, message):
        if self.progress is not None:
            self.progress(
                WorkflowProgressEvent(
                    sequence=1,
                    status=WorkflowProgressStatus.ROUTE_SELECTED,
                    route="reasoning",
                    completed_stage_count=0,
                    expected_stage_count=2,
                )
            )
            self.progress(
                WorkflowProgressEvent(
                    sequence=2,
                    status=WorkflowProgressStatus.STAGE_STARTED,
                    route="reasoning",
                    stage_id="reasoning",
                    completed_stage_count=0,
                    expected_stage_count=2,
                )
            )
        if self.error is not None:
            raise self.error
        return RAGResponse(
            content=f"Grounded: {message}",
            citations=(
                Citation(
                    position=1,
                    document_id="doc_architecture",
                    chunk_id="chunk_architecture",
                    source="/private/path/architecture.pdf",
                    score=0.91,
                    page=2,
                ),
            ),
        )


def make_runtime_resources(tmp_path):
    return AssistantRuntimeResources(
        chunker=TextChunker(chunk_size=80, overlap=10),
        retriever=HybridRetriever(
            embedding_client=DeterministicEmbeddingClient(),
            vector_store=InMemoryVectorStore(),
        ),
        memory_store=JSONMemoryStore(tmp_path / "memory.json"),
    )


def configure_providers(monkeypatch):
    for provider in ("ANTHROPIC", "OPENAI", "GEMINI"):
        monkeypatch.setenv(f"{provider}_API_KEY", f"{provider.lower()}-secret")
        monkeypatch.setenv(f"{provider}_MODEL", f"{provider.lower()}-model")


def make_service(tmp_path, **kwargs):
    return KnowledgeAssistantService(
        runtime=make_runtime_resources(tmp_path),
        ingestor=create_default_ingestor(),
        upload_dir=tmp_path / "uploads",
        **kwargs,
    )


def test_service_indexes_lists_replaces_and_deletes_upload(tmp_path):
    service = make_service(tmp_path)

    first = service.index_upload("Architecture Guide.MD", b"Routing facts")
    second = service.index_upload("Architecture Guide.MD", b"Updated routing")

    assert first["document_id"] == second["document_id"]
    assert second["source"] == "Architecture-Guide.md"
    assert second["format"] == "md"
    assert len(service.documents()) == 1
    assert service.delete_document(str(second["document_id"])) > 0
    assert service.documents() == []
    assert not (tmp_path / "uploads" / "Architecture-Guide.md").exists()


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("../secret.txt", "path"),
        ("folder\\secret.txt", "path"),
        ("document.exe", "Supported"),
        ("---.txt", "invalid"),
    ],
)
def test_service_rejects_unsafe_upload_names(tmp_path, filename, message):
    service = make_service(tmp_path)

    with pytest.raises(UploadValidationError, match=message):
        service.index_upload(filename, b"content")


def test_service_rejects_empty_and_oversized_uploads(tmp_path):
    service = make_service(tmp_path, upload_max_bytes=4)

    with pytest.raises(UploadValidationError, match="empty"):
        service.index_upload("guide.txt", b"")
    with pytest.raises(UploadValidationError, match="exceeds"):
        service.index_upload("guide.txt", b"12345")


def test_service_streams_adaptive_progress_answer_and_safe_citation(
    tmp_path,
    monkeypatch,
):
    configure_providers(monkeypatch)
    service = make_service(
        tmp_path,
        adaptive_multi_model_manager_factory=lambda progress: FakeManager(progress),
    )

    events = list(
        service.start_chat(
            ChatStreamRequest(
                message="Explain routing",
                mode="adaptive",
            )
        ).events
    )

    assert [event["event"] for event in events] == [
        "run",
        "route",
        "stage",
        "citations",
        "answer",
    ]
    citation = events[-2]["data"][0]
    assert citation["source"] == "architecture.pdf"
    assert citation["page"] == 2
    assert events[-1]["data"]["content"] == "Grounded: Explain routing"


def test_service_contains_untrusted_provider_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-openai-value")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    service = make_service(
        tmp_path,
        single_model_manager_factory=lambda _: FakeManager(
            error=RuntimeError("request included private-openai-value")
        ),
    )

    events = list(
        service.start_chat(
            ChatStreamRequest(
                message="Question",
                mode="single",
                provider="openai",
            )
        ).events
    )

    assert events[-1]["event"] == "error"
    assert "private-openai-value" not in str(events[-1])


def test_service_cancels_before_manager_start_and_rejects_concurrency(
    tmp_path,
    monkeypatch,
):
    configure_providers(monkeypatch)
    release_factory = Event()

    def blocking_factory(progress):
        release_factory.wait(timeout=2)
        return FakeManager(progress)

    service = make_service(
        tmp_path,
        adaptive_multi_model_manager_factory=blocking_factory,
    )
    run = service.start_chat(
        ChatStreamRequest(
            message="Long request",
            mode="adaptive",
        )
    )

    with pytest.raises(ActiveRunError, match="already running"):
        service.start_chat(
            ChatStreamRequest(
                message="Second request",
                mode="adaptive",
            )
        )
    assert service.cancel(run.run_id) is True
    release_factory.set()
    events = list(run.events)
    assert events[-1]["event"] == "cancelled"


def test_single_model_cancellation_suppresses_a_late_answer(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "private-openai-value")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    request_started = Event()
    release_request = Event()

    class BlockingManager:
        client = object()

        def send_message_with_citations(self, message):
            request_started.set()
            release_request.wait(timeout=2)
            return RAGResponse(f"Late: {message}", ())

    service = make_service(
        tmp_path,
        single_model_manager_factory=lambda _: BlockingManager(),
    )
    run = service.start_chat(
        ChatStreamRequest(
            message="Long request",
            mode="single",
            provider="openai",
        )
    )

    assert request_started.wait(timeout=1)
    assert service.cancel(run.run_id) is True
    release_request.set()

    events = list(run.events)
    assert [event["event"] for event in events] == ["run", "cancelled"]


def test_service_status_and_reset_never_return_secret_values(
    tmp_path,
    monkeypatch,
):
    configure_providers(monkeypatch)
    adaptive_file = tmp_path / "adaptive.json"
    monkeypatch.setattr(service_module, "ADAPTIVE_CONVERSATION_FILE", adaptive_file)
    service = make_service(tmp_path)

    status = service.status()
    service.reset_conversation(ConversationResetRequest(mode="adaptive"))

    assert status["adaptive_ready"] is True
    assert "secret" not in str(status)
    assert adaptive_file.exists()
