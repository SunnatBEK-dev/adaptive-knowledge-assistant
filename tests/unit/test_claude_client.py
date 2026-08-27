from types import SimpleNamespace

import pytest

import ai_sdk.llm.claude as claude_module
from ai_sdk.llm.claude import ClaudeClient


class FakeStream:
    def __init__(self, chunks):
        self.text_stream = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeMessagesAPI:
    def __init__(self):
        self.create_call = None
        self.stream_call = None

    def create(self, **kwargs):
        self.create_call = kwargs
        return SimpleNamespace(content=[
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="tool_use", name="ignored"),
            SimpleNamespace(type="text", text="world"),
        ])

    def stream(self, **kwargs):
        self.stream_call = kwargs
        return FakeStream(["one", "two"])


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeMessagesAPI()


def test_ask_uses_injected_client_and_provider_configuration():
    fake = FakeAnthropicClient()
    client = ClaudeClient(
        client=fake,
        model="claude-test",
        max_tokens=77,
    )
    messages = [{"role": "user", "content": "Hi"}]

    response = client.ask(messages)

    assert response == "Hello world"
    assert fake.messages.create_call == {
        "model": "claude-test",
        "max_tokens": 77,
        "messages": messages,
    }


def test_stream_uses_injected_client_without_network_access():
    fake = FakeAnthropicClient()
    client = ClaudeClient(
        client=fake,
        model="claude-test",
        max_tokens=55,
    )
    messages = [{"role": "user", "content": "Hi"}]

    assert list(client.stream(messages)) == ["one", "two"]
    assert fake.messages.stream_call == {
        "model": "claude-test",
        "max_tokens": 55,
        "messages": messages,
    }


def test_missing_model_fails_before_request(monkeypatch):
    monkeypatch.setattr(claude_module, "MODEL", None)

    with pytest.raises(RuntimeError, match="MODEL"):
        ClaudeClient(client=FakeAnthropicClient())


def test_missing_api_key_fails_before_real_client_is_created(monkeypatch):
    monkeypatch.setattr(claude_module, "API_KEY", None)
    monkeypatch.setattr(claude_module, "MODEL", "claude-test")

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeClient()
