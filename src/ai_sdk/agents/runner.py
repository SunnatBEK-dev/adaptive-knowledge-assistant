from collections.abc import Callable

from ai_sdk.agents.model import (
    AgentEvent,
    AgentModelResponse,
    AgentStopReason,
)
from ai_sdk.agents.state import AgentState
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.types import LLMMessage
from ai_sdk.observability import (
    TraceCategory,
    Tracer,
    trace_span,
)
from ai_sdk.tools.executor import ToolExecutor


AgentEventHandler = Callable[[AgentEvent], None]


class AgentRunner:
    """Provider-neutral policy for bounded tool-assisted runs."""

    def __init__(
        self,
        client: BaseToolLLMClient,
        executor: ToolExecutor,
        *,
        max_tool_rounds: int = 8,
        tracer: Tracer | None = None,
    ) -> None:
        if not isinstance(client, BaseToolLLMClient):
            raise TypeError(
                "Agent client must support tool turns."
            )

        if not isinstance(executor, ToolExecutor):
            raise TypeError(
                "Tool executor must be a ToolExecutor."
            )

        if (
            not isinstance(max_tool_rounds, int)
            or isinstance(max_tool_rounds, bool)
            or max_tool_rounds <= 0
        ):
            raise ValueError(
                "Maximum tool rounds must be greater than zero."
            )
        if tracer is not None and not isinstance(tracer, Tracer):
            raise TypeError("Agent tracer must be a Tracer.")

        self.client = client
        self.executor = executor
        self.max_tool_rounds = max_tool_rounds
        self.tracer = tracer or executor.tracer

    def run(
        self,
        messages: list[LLMMessage],
        *,
        on_event: AgentEventHandler | None = None,
    ) -> AgentState:
        with trace_span(
            self.tracer,
            "agent.run",
            TraceCategory.AGENT,
            {
                "agent.message_count": len(messages),
                "agent.max_tool_rounds": self.max_tool_rounds,
            },
        ) as span:
            state = self._run(messages, on_event=on_event)
            if span is not None:
                span.set_attribute(
                    "agent.tool_round_count",
                    state.tool_rounds,
                )
                span.set_attribute(
                    "agent.stop_reason",
                    state.stop_reason.value,
                )
                if (
                    state.stop_reason
                    is AgentStopReason.MAX_TOOL_ROUNDS
                ):
                    span.set_error("MaxToolRoundsExceeded")
            return state

    def _run(
        self,
        messages: list[LLMMessage],
        *,
        on_event: AgentEventHandler | None = None,
    ) -> AgentState:
        if on_event is not None and not callable(on_event):
            raise TypeError("Agent event handler must be callable.")

        state = AgentState(messages)
        schemas = self.executor.registry.schemas()
        seen_call_ids: set[str] = set()

        while True:
            with trace_span(
                self.tracer,
                "llm.tool_turn",
                TraceCategory.LLM,
                {
                    "llm.message_count": len(state.messages),
                    "llm.tool_schema_count": len(schemas),
                    "llm.prior_event_count": len(state.events),
                },
            ) as model_span:
                response = self.client.complete_tool_turn(
                    state.messages,
                    schemas,
                    tuple(state.events),
                )
                if model_span is not None and isinstance(
                    response,
                    AgentModelResponse,
                ):
                    model_span.set_attribute(
                        "llm.tool_call_count",
                        len(response.tool_calls),
                    )

            if not isinstance(response, AgentModelResponse):
                raise TypeError(
                    "Agent client response is invalid."
                )

            calls = response.tool_calls
            call_ids = [call.id for call in calls]

            if (
                len(call_ids) != len(set(call_ids))
                or any(
                    call_id in seen_call_ids
                    for call_id in call_ids
                )
            ):
                raise RuntimeError(
                    "Agent received a duplicate tool call ID."
                )

            if calls and state.tool_rounds >= self.max_tool_rounds:
                event = AgentEvent(
                    iteration=len(state.events) + 1,
                    response=response,
                )
                self._record(state, event, on_event)
                state.finish(
                    AgentStopReason.MAX_TOOL_ROUNDS,
                    response.text,
                )
                return state

            results = tuple(
                self.executor.execute(call, tracer=self.tracer)
                for call in calls
            )
            event = AgentEvent(
                iteration=len(state.events) + 1,
                response=response,
                tool_results=results,
            )
            self._record(state, event, on_event)

            if not calls:
                state.finish(
                    AgentStopReason.FINAL_RESPONSE,
                    response.text,
                )
                return state

            seen_call_ids.update(call_ids)

    def ask(
        self,
        messages: list[LLMMessage],
        *,
        on_event: AgentEventHandler | None = None,
    ) -> str:
        state = self.run(messages, on_event=on_event)

        if state.stop_reason is AgentStopReason.MAX_TOOL_ROUNDS:
            raise RuntimeError(
                "Maximum agent tool rounds exceeded."
            )

        return state.final_text or ""

    @staticmethod
    def _record(
        state: AgentState,
        event: AgentEvent,
        handler: AgentEventHandler | None,
    ) -> None:
        state.record(event)

        if handler is not None:
            handler(event)
