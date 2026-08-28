from hashlib import sha256

import pytest

from app.main import (
    build_direct_chat_manager,
    build_manager,
    build_super_ai_manager,
    load_document,
    run_cli,
    select_application_mode,
    select_direct_provider,
)
import app.main as main_module
from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTextBlock,
    AgentWorker,
    DependencyHandoffCoordinator,
    HandoffOutputFormat,
    SuperAIRoute,
)
from ai_sdk.application import ApplicationMode
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.super_ai import (
    RoutedSuperAIClient,
    SuperAIClient,
)
from ai_sdk.tools import ToolExecutor, ToolRegistry


def test_load_document_uses_stable_path_identity(tmp_path):
    file_path = tmp_path / "Python guide.txt"
    file_path.write_text(
        "Python functions",
        encoding="utf-8",
    )

    first = load_document(str(file_path))
    file_path.write_text(
        "Updated Python functions",
        encoding="utf-8",
    )
    second = load_document(str(file_path))

    assert first.id == second.id
    assert first.id.startswith("doc_")
    assert first.content == "Python functions"
    assert second.content == "Updated Python functions"
    assert second.metadata == {
        "source": str(file_path.resolve()),
        "format": "txt",
        "content_hash": sha256(
            b"Updated Python functions"
        ).hexdigest(),
    }


def test_load_document_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_document(str(tmp_path / "missing.txt"))


def test_load_document_rejects_non_utf8_file(tmp_path):
    file_path = tmp_path / "binary.txt"
    file_path.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="UTF-8"):
        load_document(str(file_path))


def test_cli_selects_application_mode_after_invalid_choice(capsys):
    choices = iter(["unknown", "2"])

    mode = select_application_mode(lambda _: next(choices))

    assert mode is ApplicationMode.SUPER_AI
    assert "Invalid mode" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("selection", "provider"),
    [
        ("1", "anthropic"),
        ("Claude", "anthropic"),
        ("2", "openai"),
        ("GPT", "openai"),
        ("3", "gemini"),
        ("Gemini", "gemini"),
    ],
)
def test_cli_selects_direct_chat_provider(selection, provider):
    assert select_direct_provider(lambda _: selection) == provider


def test_cli_reprompts_for_invalid_provider(capsys):
    choices = iter(["other", "openai"])

    provider = select_direct_provider(lambda _: next(choices))

    assert provider == "openai"
    assert "Invalid provider" in capsys.readouterr().out


class NonStreamingManager:
    def __init__(self):
        self.prompts = []
        self.last_citations = ()

    def send_message(self, prompt):
        self.prompts.append(prompt)
        return "Combined answer"


def test_cli_can_run_non_streaming_super_ai_mode(capsys):
    manager = NonStreamingManager()
    commands = iter(["Question", "/exit"])

    run_cli(
        manager,
        input_fn=lambda _: next(commands),
        ingestor=object(),
        title="Super AI",
        stream_responses=False,
    )

    output = capsys.readouterr().out
    assert "Super AI" in output
    assert "Assistant: Combined answer" in output
    assert manager.prompts == ["Question"]


def test_manager_builder_rejects_provider_and_explicit_client():
    with pytest.raises(ValueError, match="either"):
        build_manager(provider="openai", client=object())

    with pytest.raises(TypeError, match="BaseLLMClient"):
        build_manager(client=object())


def test_direct_chat_builder_uses_normalized_provider_history(
    monkeypatch,
    tmp_path,
):
    received = {}
    expected = object()

    def fake_build_manager(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(
        main_module,
        "build_manager",
        fake_build_manager,
    )
    chat_file = tmp_path / "openai.json"

    result = build_direct_chat_manager(
        " OpenAI ",
        conversation_file=chat_file,
    )

    assert result is expected
    assert received == {
        "provider": "openai",
        "conversation_file": chat_file,
    }


class WorkerClient(BaseToolLLMClient):
    def ask(self, messages):
        return "done"

    def stream(self, messages):
        yield "done"

    def complete_tool_turn(self, messages, schemas, events):
        return AgentModelResponse([AgentTextBlock("done")])


def test_super_ai_builder_configures_three_provider_stages(
    monkeypatch,
    tmp_path,
):
    providers = []
    received = {}
    expected = object()

    def fake_worker(name, description, provider):
        providers.append(provider)
        return AgentWorker(
            name,
            description,
            AgentRunner(
                WorkerClient(),
                ToolExecutor(ToolRegistry()),
            ),
            provider=provider,
        )

    def fake_build_manager(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(
        main_module,
        "create_provider_worker",
        fake_worker,
    )
    monkeypatch.setattr(
        main_module,
        "build_manager",
        fake_build_manager,
    )
    chat_file = tmp_path / "super.json"

    result = build_super_ai_manager(
        conversation_file=chat_file
    )

    assert result is expected
    assert providers == ["gemini", "anthropic", "openai"]
    assert received["conversation_file"] == chat_file
    routed = received["client"]
    assert isinstance(routed, RoutedSuperAIClient)
    assert set(routed.workflows) == set(SuperAIRoute)
    assert {
        route: len(client.workflow.stages)
        for route, client in routed.workflows.items()
    } == {
        SuperAIRoute.FAST: 1,
        SuperAIRoute.CONTEXT: 2,
        SuperAIRoute.REASONING: 2,
        SuperAIRoute.FULL: 3,
    }
    full = routed.workflows[SuperAIRoute.FULL]
    assert isinstance(full, SuperAIClient)
    assert isinstance(
        full.workflow,
        DependencyHandoffCoordinator,
    )
    assert [
        stage.output_format
        for stage in full.workflow.stages
    ] == [
        HandoffOutputFormat.STRUCTURED,
        HandoffOutputFormat.STRUCTURED,
        HandoffOutputFormat.TEXT,
    ]
    assert [
        stage.depends_on
        for stage in full.workflow.stages
    ] == [
        (),
        ("context",),
        ("context", "reasoning"),
    ]
