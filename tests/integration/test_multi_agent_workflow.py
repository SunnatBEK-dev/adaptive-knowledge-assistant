import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTask,
    AgentTaskStatus,
    AgentTextBlock,
    AgentWorker,
    MultiAgentCoordinator,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.tools import (
    ToolCall,
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolSchema,
)


pytestmark = pytest.mark.integration


class ScriptedWorkerClient(BaseToolLLMClient):
    def __init__(self, responses):
        self.responses = list(responses)

    def ask(self, messages):
        return "unused"

    def stream(self, messages):
        yield "unused"

    def complete_tool_turn(self, messages, schemas, events):
        return self.responses.pop(0)


def test_named_workers_produce_isolated_ordered_results():
    registry = ToolRegistry()
    registry.register(
        ToolSchema(
            name="lookup",
            description="Look up a topic.",
            parameters=[
                ToolParameter(
                    "topic",
                    ToolParameterType.STRING,
                    "Topic to look up.",
                ),
            ],
        ),
        lambda topic: {"topic": topic, "fact": "verified"},
    )
    researcher = AgentWorker(
        "researcher",
        "Collect verified facts",
        AgentRunner(
            ScriptedWorkerClient([
                AgentModelResponse([
                    ToolCall(
                        "call_lookup",
                        "lookup",
                        {"topic": "Python"},
                    ),
                ]),
                AgentModelResponse([
                    AgentTextBlock("Research complete"),
                ]),
            ]),
            ToolExecutor(registry),
        ),
    )
    writer = AgentWorker(
        "writer",
        "Write concise text",
        AgentRunner(
            ScriptedWorkerClient([
                AgentModelResponse([
                    AgentTextBlock("Draft complete"),
                ]),
            ]),
            ToolExecutor(ToolRegistry()),
        ),
    )
    coordinator = MultiAgentCoordinator([
        researcher,
        writer,
    ])

    result = coordinator.run([
        AgentTask(
            "task_research",
            "researcher",
            "Research Python",
        ),
        AgentTask(
            "task_write",
            "writer",
            "Write a short draft",
        ),
    ])

    assert [item.status for item in result.results] == [
        AgentTaskStatus.COMPLETED,
        AgentTaskStatus.COMPLETED,
    ]
    assert [item.output for item in result.results] == [
        "Research complete",
        "Draft complete",
    ]
    assert result.results[0].state.tool_rounds == 1
    assert result.results[1].state.tool_rounds == 0
    assert (
        result.results[0].state
        is not result.results[1].state
    )
