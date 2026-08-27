import pytest

from ai_sdk.application.conversation_manager import ConversationManager
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation


class FakeClient:
    def __init__(
        self,
        response="Assistant response",
        chunks=None,
        error=None,
    ):
        self.response = response
        self.chunks = chunks or ["Assistant ", "response"]
        self.error = error
        self.received_messages = None

    def ask(self, messages):
        self.received_messages = messages
        if self.error:
            raise self.error
        return self.response

    def stream(self, messages):
        self.received_messages = messages
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


class FakeRepository:
    def __init__(self, error=None):
        self.error = error
        self.saved = []

    def save(self, conversation):
        if self.error:
            raise self.error
        self.saved.append([
            message.to_dict()
            for message in conversation.history()
        ])

    def load(self):
        return Conversation()


def build_manager(client=None, repository=None):
    conversation = Conversation()
    client = client or FakeClient()
    repository = repository or FakeRepository()
    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=repository,
    )
    return manager, conversation, client, repository


def test_send_message_orchestrates_and_persists_conversation():
    manager, conversation, client, repository = build_manager()

    response = manager.send_message("User question")

    assert response == "Assistant response"
    assert client.received_messages == [
        {"role": "user", "content": "User question"},
    ]
    assert [message.content for message in conversation.history()] == [
        "User question",
        "Assistant response",
    ]
    assert len(repository.saved) == 1


def test_send_message_rolls_back_new_state_when_client_fails():
    manager, conversation, _, repository = build_manager(
        client=FakeClient(error=RuntimeError("LLM failed")),
    )
    existing = conversation.add_user("Existing")

    with pytest.raises(RuntimeError, match="LLM failed"):
        manager.send_message("New")

    assert conversation.history() == [existing]
    assert repository.saved == []


def test_send_message_rolls_back_new_state_when_save_fails():
    manager, conversation, _, _ = build_manager(
        repository=FakeRepository(error=OSError("save failed")),
    )
    existing = conversation.add_user("Existing")

    with pytest.raises(OSError, match="save failed"):
        manager.send_message("New")

    assert conversation.history() == [existing]


def test_stream_message_yields_chunks_and_persists_full_response():
    manager, conversation, client, repository = build_manager(
        client=FakeClient(chunks=["A", "B", "C"]),
    )

    chunks = list(manager.stream_message("Question"))

    assert chunks == ["A", "B", "C"]
    assert client.received_messages == [
        {"role": "user", "content": "Question"},
    ]
    assert conversation.last_message().content == "ABC"
    assert len(repository.saved) == 1


def test_stream_message_rolls_back_when_stream_fails():
    manager, conversation, _, repository = build_manager(
        client=FakeClient(
            chunks=["partial"],
            error=RuntimeError("stream failed"),
        ),
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        list(manager.stream_message("Question"))

    assert conversation.is_empty()
    assert repository.saved == []


def test_stream_message_rolls_back_when_consumer_closes_early():
    manager, conversation, _, repository = build_manager(
        client=FakeClient(chunks=["first", "second"]),
    )
    stream = manager.stream_message("Question")

    assert next(stream) == "first"
    stream.close()

    assert conversation.is_empty()
    assert repository.saved == []


def test_stream_message_rolls_back_when_save_fails():
    manager, conversation, _, _ = build_manager(
        repository=FakeRepository(error=OSError("save failed")),
    )

    with pytest.raises(OSError, match="save failed"):
        list(manager.stream_message("Question"))

    assert conversation.is_empty()
