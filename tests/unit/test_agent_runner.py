import pytest

from ai_sdk.agents import (
    AgentEvent,
    AgentModelResponse,
    AgentRunner,
    AgentState,
    AgentStopReason,
    AgentTextBlock,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.observability import (
    InMemoryTraceCollector,
    TraceStatus,
    Tracer,
)
from ai_sdk.tools import (
    ToolCall,
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolRegistry,
    ToolResult,
    ToolSchema,
)


class ScriptedAgentClient(BaseToolLLMClient):
    def __init__(self, responses):
        self.responses = list(responses)
        self.turns = []

    def ask(self, messages):
        return "plain"

    def stream(self, messages):
        yield "plain"

    def complete_tool_turn(self, messages, schemas, events):
        self.turns.append((messages, schemas, events))
        return self.responses.pop(0)


def build_executor(handler=lambda value: value * 2):
    registry = ToolRegistry()
    registry.register(
        ToolSchema(
            name="double",
            description="Double an integer.",
            parameters=[
                ToolParameter(
                    "value",
                    ToolParameterType.INTEGER,
                    "Integer to double.",
                ),
            ],
        ),
        handler,
    )
    return ToolExecutor(registry)


def tool_response(call_id, value):
    return AgentModelResponse([
        AgentTextBlock("Calculating."),
        ToolCall(
            id=call_id,
            name="double",
            arguments={"value": value},
        ),
    ])


def final_response(text="Final answer"):
    return AgentModelResponse([AgentTextBlock(text)])


def test_agent_response_preserves_block_order_and_views():
    call = ToolCall("call_1", "double", {"value": 2})
    response = AgentModelResponse([
        AgentTextBlock("Before "),
        call,
        AgentTextBlock("after."),
    ])

    assert response.blocks == (
        AgentTextBlock("Before "),
        call,
        AgentTextBlock("after."),
    )
    assert response.text == "Before after."
    assert response.tool_calls == (call,)


def test_agent_event_validates_iteration_and_matching_results():
    response = tool_response("call_1", 2)
    result = ToolResult("call_1", "double", "4")

    event = AgentEvent(1, response, [result])

    assert event.tool_results == (result,)

    with pytest.raises(ValueError, match="iteration"):
        AgentEvent(0, response)

    with pytest.raises(ValueError, match="do not match"):
        AgentEvent(
            1,
            response,
            [ToolResult("other", "double", "4")],
        )


def test_agent_state_copies_messages_and_enforces_lifecycle():
    messages = [{"role": "user", "content": "Original"}]
    state = AgentState(messages)
    event = AgentEvent(1, final_response())

    messages[0]["content"] = "Changed"
    state.record(event)
    state.finish(
        AgentStopReason.FINAL_RESPONSE,
        "Final answer",
    )

    assert state.messages[0]["content"] == "Original"
    assert state.events == [event]
    assert state.is_finished is True
    assert state.tool_rounds == 0

    with pytest.raises(RuntimeError, match="cannot record"):
        state.record(AgentEvent(2, final_response()))

    with pytest.raises(RuntimeError, match="already finished"):
        state.finish(
            AgentStopReason.FINAL_RESPONSE,
            "Again",
        )


def test_runner_returns_final_state_and_emits_event():
    client = ScriptedAgentClient([final_response("Done")])
    emitted = []
    runner = AgentRunner(client, build_executor())
    messages = [{"role": "user", "content": "Question"}]

    state = runner.run(messages, on_event=emitted.append)

    assert state.stop_reason is AgentStopReason.FINAL_RESPONSE
    assert state.final_text == "Done"
    assert len(state.events) == 1
    assert emitted == state.events
    assert client.turns[0][0] == messages
    assert len(client.turns[0][1]) == 1
    assert client.turns[0][2] == ()


def test_runner_executes_tool_and_passes_event_to_next_turn():
    executions = []
    client = ScriptedAgentClient([
        tool_response("call_1", 3),
        final_response("Six"),
    ])
    executor = build_executor(
        lambda value: executions.append(value) or value * 2
    )

    state = AgentRunner(client, executor).run([
        {"role": "user", "content": "Double three."},
    ])

    assert state.stop_reason is AgentStopReason.FINAL_RESPONSE
    assert state.final_text == "Six"
    assert state.tool_rounds == 1
    assert executions == [3]
    assert state.events[0].tool_results == (
        ToolResult("call_1", "double", "6"),
    )
    assert client.turns[1][2] == (state.events[0],)


def test_runner_traces_nested_model_and_tool_operations():
    collector = InMemoryTraceCollector()
    tracer = Tracer(collector)
    client = ScriptedAgentClient([
        tool_response("call_1", 3),
        final_response("Six"),
    ])

    state = AgentRunner(
        client,
        build_executor(),
        tracer=tracer,
    ).run([{"role": "user", "content": "private prompt"}])

    records = collector.records()
    root = next(record for record in records if record.name == "agent.run")
    children = [
        record
        for record in records
        if record.parent_span_id == root.span_id
    ]
    assert state.final_text == "Six"
    assert [record.name for record in records].count(
        "llm.tool_turn"
    ) == 2
    assert any(record.name == "tool.execute" for record in children)
    assert all(record.trace_id == root.trace_id for record in records)
    assert root.attributes["agent.tool_round_count"] == 1
    assert root.attributes["agent.stop_reason"] == "final_response"
    assert root.status is TraceStatus.OK
    assert "private prompt" not in str(
        [record.to_dict() for record in records]
    )


def test_runner_returns_explicit_max_round_stop_without_extra_execution():
    executions = []
    collector = InMemoryTraceCollector()
    client = ScriptedAgentClient([
        tool_response("call_1", 1),
        tool_response("call_2", 2),
    ])
    executor = build_executor(
        lambda value: executions.append(value) or value * 2
    )

    state = AgentRunner(
        client,
        executor,
        max_tool_rounds=1,
        tracer=Tracer(collector),
    ).run([{"role": "user", "content": "Keep going."}])

    assert state.stop_reason is AgentStopReason.MAX_TOOL_ROUNDS
    assert state.tool_rounds == 1
    assert len(state.events) == 2
    assert state.events[-1].tool_results == ()
    assert executions == [1]
    root = next(
        record
        for record in collector.records()
        if record.name == "agent.run"
    )
    assert root.status is TraceStatus.ERROR
    assert root.error_type == "MaxToolRoundsExceeded"


def test_runner_ask_raises_when_max_rounds_are_reached():
    client = ScriptedAgentClient([
        tool_response("call_1", 1),
        tool_response("call_2", 2),
    ])
    runner = AgentRunner(
        client,
        build_executor(),
        max_tool_rounds=1,
    )

    with pytest.raises(RuntimeError, match="rounds exceeded"):
        runner.ask([
            {"role": "user", "content": "Keep going."},
        ])


@pytest.mark.parametrize(
    "responses",
    [
        [
            AgentModelResponse([
                ToolCall("same", "double", {"value": 1}),
                ToolCall("same", "double", {"value": 2}),
            ]),
        ],
        [
            tool_response("same", 1),
            tool_response("same", 2),
        ],
    ],
)
def test_runner_rejects_duplicate_call_ids(responses):
    runner = AgentRunner(
        ScriptedAgentClient(responses),
        build_executor(),
    )

    with pytest.raises(RuntimeError, match="duplicate"):
        runner.run([{"role": "user", "content": "Run."}])


@pytest.mark.parametrize("max_tool_rounds", [0, -1, True, 1.5])
def test_runner_rejects_invalid_max_rounds(max_tool_rounds):
    with pytest.raises(ValueError, match="greater than zero"):
        AgentRunner(
            ScriptedAgentClient([final_response()]),
            build_executor(),
            max_tool_rounds=max_tool_rounds,
        )


def test_runner_rejects_invalid_dependencies_and_outputs():
    with pytest.raises(TypeError, match="support tool turns"):
        AgentRunner(object(), build_executor())

    with pytest.raises(TypeError, match="ToolExecutor"):
        AgentRunner(
            ScriptedAgentClient([final_response()]),
            object(),
        )

    with pytest.raises(TypeError, match="tracer"):
        AgentRunner(
            ScriptedAgentClient([final_response()]),
            build_executor(),
            tracer=object(),
        )

    runner = AgentRunner(
        ScriptedAgentClient(["not-a-response"]),
        build_executor(),
    )

    with pytest.raises(TypeError, match="response is invalid"):
        runner.run([{"role": "user", "content": "Run."}])

    valid = AgentRunner(
        ScriptedAgentClient([final_response()]),
        build_executor(),
    )

    with pytest.raises(TypeError, match="handler"):
        valid.run(
            [{"role": "user", "content": "Run."}],
            on_event=object(),
        )
