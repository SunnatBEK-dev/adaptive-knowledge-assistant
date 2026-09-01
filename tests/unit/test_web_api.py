import json

from fastapi.testclient import TestClient

from ai_sdk.application.bootstrap import AssistantRuntimeResources
from ai_sdk.application.rag_response import RAGResponse
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.memory.json_store import JSONMemoryStore
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.hybrid import HybridRetriever
from ai_sdk.retrieval.in_memory import InMemoryVectorStore
from ai_sdk.web.app import _sse_events, create_app
from ai_sdk.web.service import KnowledgeAssistantService


class TinyEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class AnswerManager:
    client = object()

    def send_message_with_citations(self, message):
        return RAGResponse(f"Answer: {message}", ())


def make_client(tmp_path, monkeypatch, *, upload_limit=1024):
    monkeypatch.setenv("OPENAI_API_KEY", "never-return-this-secret")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    runtime = AssistantRuntimeResources(
        chunker=TextChunker(chunk_size=80, overlap=10),
        retriever=HybridRetriever(
            embedding_client=TinyEmbeddingClient(),
            vector_store=InMemoryVectorStore(),
        ),
        memory_store=JSONMemoryStore(tmp_path / "memory.json"),
    )
    service = KnowledgeAssistantService(
        runtime=runtime,
        upload_dir=tmp_path / "uploads",
        upload_max_bytes=upload_limit,
        single_model_manager_factory=lambda _: AnswerManager(),
    )
    return TestClient(create_app(service))


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ")
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        events.append((event_type, data))
    return events


def test_web_home_status_and_security_headers(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    home = client.get("/")
    status = client.get("/api/status")

    assert home.status_code == 200
    assert "Adaptive Knowledge Assistant" in home.text
    assert "/static/favicon.svg" in home.text
    assert "default-src 'self'" in home.headers["content-security-policy"]
    assert status.status_code == 200
    assert status.json()["providers"][1]["ready"] is True
    assert "never-return-this-secret" not in status.text


def test_document_upload_list_delete_and_validation(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch, upload_limit=12)

    uploaded = client.post(
        "/api/documents",
        files={"file": ("Guide.md", b"RAG facts", "text/markdown")},
    )
    document = uploaded.json()["document"]

    assert uploaded.status_code == 201
    assert client.get("/api/documents").json()[0]["source"] == "Guide.md"
    assert client.delete(f"/api/documents/{document['document_id']}").status_code == 204
    assert client.delete(f"/api/documents/{document['document_id']}").status_code == 404

    unsafe = client.post(
        "/api/documents",
        files={"file": ("../private.txt", b"text", "text/plain")},
    )
    oversized = client.post(
        "/api/documents",
        files={"file": ("large.txt", b"x" * 13, "text/plain")},
    )
    assert unsafe.status_code == 400
    assert oversized.status_code == 413


def test_chat_sse_validation_and_reset(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/chat/stream",
        json={
            "message": "Explain hybrid retrieval",
            "mode": "single",
            "provider": "openai",
        },
    )
    events = parse_sse(response.text)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-run-id"].startswith("run_")
    assert [event[0] for event in events] == ["run", "citations", "answer"]
    assert events[-1][1]["content"] == "Answer: Explain hybrid retrieval"

    invalid = client.post(
        "/api/chat/stream",
        json={"message": "Question", "mode": "single"},
    )
    reset = client.post(
        "/api/conversations/reset",
        json={"mode": "single", "provider": "openai"},
    )
    assert invalid.status_code == 422
    assert reset.status_code == 200
    assert reset.json() == {"reset": True}


def test_sse_disconnect_closes_the_service_event_stream():
    closed = []

    def source():
        try:
            yield {"event": "run", "data": {"run_id": "run_one"}}
            yield {"event": "answer", "data": {"content": "late"}}
        finally:
            closed.append(True)

    stream = _sse_events(source())

    assert next(stream).startswith("event: run")
    stream.close()

    assert closed == [True]
