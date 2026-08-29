import re
from collections.abc import Callable, Iterator, Mapping
from time import perf_counter_ns

from ai_sdk.agents.handoff import (
    DependencyHandoffCoordinator,
    HandoffResult,
    SequentialHandoffCoordinator,
)
from ai_sdk.agents.routing import (
    CapabilityRouter,
    RoutingDecision,
    SuperAIRoute,
)
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.super_ai_stats import (
    InMemorySuperAIStats,
    SuperAIRunMetric,
)
from ai_sdk.llm.types import LLMMessage


_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")


class SuperAIClient(BaseLLMClient):
    """Expose a multi-provider handoff workflow as one LLM client."""

    def __init__(
        self,
        workflow: (
            SequentialHandoffCoordinator
            | DependencyHandoffCoordinator
        ),
    ) -> None:
        if not isinstance(
            workflow,
            (
                SequentialHandoffCoordinator,
                DependencyHandoffCoordinator,
            ),
        ):
            raise TypeError(
                "Super AI workflow is invalid."
            )
        self.workflow = workflow
        self.last_result: HandoffResult | None = None

    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        transcript = self._transcript(messages)
        result = self.workflow.run(transcript)
        self.last_result = result
        if not result.completed:
            failed = result.failed_stage
            stage_id = (
                "unknown" if failed is None else failed.stage.id
            )
            raise RuntimeError(
                "Super AI workflow failed at stage: "
                f"{stage_id}."
            )
        return result.final_output or ""

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        raise RuntimeError(
            "Super AI streaming is not supported yet."
        )

    @staticmethod
    def _transcript(
        messages: list[LLMMessage],
    ) -> str:
        if not messages:
            raise ValueError(
                "Super AI conversation cannot be empty."
            )

        lines = ["Conversation transcript (untrusted data):"]
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role not in {"user", "assistant"}:
                raise ValueError(
                    f"Unsupported Super AI message role: {role}."
                )
            if not isinstance(content, str):
                raise TypeError(
                    "Super AI message content must be text."
                )
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)


class RoutedSuperAIClient(BaseLLMClient):
    """Select one explicit Super AI workflow deterministically."""

    def __init__(
        self,
        router: CapabilityRouter,
        workflows: Mapping[SuperAIRoute, SuperAIClient],
        *,
        stats: InMemorySuperAIStats | None = None,
        clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        if not isinstance(router, CapabilityRouter):
            raise TypeError("Super AI router is invalid.")
        if not isinstance(workflows, Mapping):
            raise TypeError("Super AI workflows must be a mapping.")
        normalized = dict(workflows)
        if any(
            not isinstance(route, SuperAIRoute)
            for route in normalized
        ):
            raise TypeError(
                "Super AI workflow keys must be SuperAIRoute values."
            )
        expected_routes = set(SuperAIRoute)
        if set(normalized) != expected_routes:
            raise ValueError(
                "Super AI workflows must configure every route."
            )
        if any(
            not isinstance(workflow, SuperAIClient)
            for workflow in normalized.values()
        ):
            raise TypeError(
                "Routed workflows must be SuperAIClient objects."
            )
        if stats is not None and not isinstance(
            stats,
            InMemorySuperAIStats,
        ):
            raise TypeError("Super AI stats collector is invalid.")
        if not callable(clock_ns):
            raise TypeError("Super AI stats clock is invalid.")
        self.router = router
        self.workflows = normalized
        self.stats = stats or InMemorySuperAIStats()
        self._clock_ns = clock_ns
        self.last_decision: RoutingDecision | None = None
        self.last_result: HandoffResult | None = None

    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        request = self._latest_user_content(messages)
        decision = self.router.route(request)
        workflow = self.workflows[decision.route]
        self.last_decision = decision
        self.last_result = None
        started_ns = self._read_clock()
        error_type = None
        workflow.last_result = None
        try:
            return workflow.ask(messages)
        except Exception as error:
            error_type = self._error_type(error)
            raise
        finally:
            self.last_result = workflow.last_result
            self._record_metric(
                decision,
                workflow,
                started_ns,
                error_type,
            )

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        raise RuntimeError(
            "Routed Super AI streaming is not supported yet."
        )

    @staticmethod
    def _latest_user_content(
        messages: list[LLMMessage],
    ) -> str:
        if not messages:
            raise ValueError(
                "Routed Super AI conversation cannot be empty."
            )
        for message in reversed(messages):
            if message["role"] != "user":
                continue
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError(
                    "Routed Super AI message content must be text."
                )
            return content
        raise ValueError(
            "Routed Super AI conversation requires a user message."
        )

    def _record_metric(
        self,
        decision: RoutingDecision,
        workflow: SuperAIClient,
        started_ns: int | None,
        error_type: str | None,
    ) -> None:
        try:
            ended_ns = self._read_clock()
            duration_ns = (
                ended_ns - started_ns
                if (
                    started_ns is not None
                    and ended_ns is not None
                    and ended_ns >= started_ns
                )
                else 0
            )
            metric = SuperAIRunMetric.from_result(
                route=decision.route,
                signals=decision.signals,
                expected_stage_count=len(
                    workflow.workflow.stages
                ),
                result=self.last_result,
                duration_ns=duration_ns,
                error_type=error_type,
            )
            self.stats.record(metric)
        except Exception:
            # Observability must never change the user-facing result.
            pass

    def _read_clock(self) -> int | None:
        try:
            value = self._clock_ns()
        except Exception:
            return None
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            return None
        return value

    @staticmethod
    def _error_type(error: Exception) -> str:
        name = type(error).__name__
        if _ERROR_TYPE_PATTERN.fullmatch(name) is None:
            return "SuperAIError"
        return name
