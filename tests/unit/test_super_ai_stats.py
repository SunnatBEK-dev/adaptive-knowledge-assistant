import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTextBlock,
    AgentWorker,
    CapabilityRouter,
    HandoffStage,
    MultiAgentCoordinator,
    RoutingSignal,
    SequentialHandoffCoordinator,
    SuperAIRoute,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.super_ai import (
    RoutedSuperAIClient,
    SuperAIClient,
)
from ai_sdk.llm.super_ai_stats import (
    InMemorySuperAIStats,
    SuperAIRunMetric,
    SuperAIStatsValidationError,
)
from ai_sdk.tools import ToolExecutor, ToolRegistry


class StatsClient(BaseToolLLMClient):
    def __init__(self, response="done", error=None):
        self.response = response
        self.error = error

    def ask(self, messages):
        return self.response

    def stream(self, messages):
        yield self.response

    def complete_tool_turn(self, messages, schemas, events):
        if self.error is not None:
            raise self.error
        return AgentModelResponse([AgentTextBlock(self.response)])


def workflow(provider):
    worker = AgentWorker(
        "worker",
        "Answer",
        AgentRunner(
            provider,
            ToolExecutor(ToolRegistry()),
        ),
    )
    return SuperAIClient(SequentialHandoffCoordinator(
        MultiAgentCoordinator([worker]),
        [HandoffStage("final", "worker", "Answer")],
    ))


def routed_client(fast_provider, *, clock_ns):
    workflows = {
        route: workflow(
            fast_provider
            if route is SuperAIRoute.FAST
            else StatsClient(route.value)
        )
        for route in SuperAIRoute
    }
    return RoutedSuperAIClient(
        CapabilityRouter(),
        workflows,
        clock_ns=clock_ns,
    )


def test_routed_client_collects_content_free_success_metrics():
    client = routed_client(
        StatsClient("private answer"),
        clock_ns=iter([1_000_000, 4_000_000]).__next__,
    )

    assert client.ask([{
        "role": "user",
        "content": "private question",
    }]) == "private answer"

    metric = client.stats.records()[0]
    assert metric.route is SuperAIRoute.FAST
    assert metric.signals == ()
    assert metric.completed is True
    assert metric.expected_stage_count == 1
    assert metric.executed_stage_ids == ("final",)
    assert metric.duration_ms == 3.0
    report = client.stats.report()
    assert report.total_runs == 1
    assert report.successful_runs == 1
    assert report.route_counts == {"fast": 1}
    assert report.stage_execution_counts == {"final": 1}
    assert "private question" not in str(report.to_dict())
    assert "private answer" not in str(report.to_dict())


def test_routed_client_records_failed_stage_without_error_message():
    client = routed_client(
        StatsClient(error=RuntimeError("private failure")),
        clock_ns=iter([10, 20]).__next__,
    )

    with pytest.raises(RuntimeError, match="stage: final"):
        client.ask([{"role": "user", "content": "hello"}])

    metric = client.stats.records()[0]
    assert metric.completed is False
    assert metric.failed_stage_ids == ("final",)
    assert metric.error_type == "RuntimeError"
    report = client.stats.report()
    assert report.failed_runs == 1
    assert report.stage_failure_counts == {"final": 1}
    assert "private failure" not in str(report.to_dict())


def test_broken_stats_clock_never_changes_business_result():
    def broken_clock():
        raise RuntimeError("clock unavailable")

    client = routed_client(StatsClient("answer"), clock_ns=broken_clock)

    assert client.ask([{
        "role": "user",
        "content": "hello",
    }]) == "answer"
    assert client.stats.records()[0].duration_ns == 0

    invalid_clock_client = routed_client(
        StatsClient("answer"),
        clock_ns=lambda: "invalid",
    )
    assert invalid_clock_client.ask([{
        "role": "user",
        "content": "hello",
    }]) == "answer"
    assert invalid_clock_client.stats.records()[0].duration_ns == 0


def test_stats_failures_and_unusual_errors_are_contained():
    class FailingStats(InMemorySuperAIStats):
        def record(self, metric):
            raise RuntimeError("stats unavailable")

    template = routed_client(StatsClient(), clock_ns=lambda: 1)
    client = RoutedSuperAIClient(
        CapabilityRouter(),
        template.workflows,
        stats=FailingStats(),
        clock_ns=lambda: 1,
    )

    assert client.ask([{
        "role": "user",
        "content": "hello",
    }]) == "done"

    unusual_error = type("invalid-error-name!", (Exception,), {})
    assert RoutedSuperAIClient._error_type(
        unusual_error("private")
    ) == "SuperAIError"


def test_stats_are_bounded_clearable_and_report_blocked_stages():
    stats = InMemorySuperAIStats(max_records=1)
    stats.record(SuperAIRunMetric(
        route=SuperAIRoute.FULL,
        signals=(RoutingSignal.LONG_REQUEST,),
        expected_stage_count=3,
        executed_stage_ids=("context",),
        failed_stage_ids=("context",),
        blocked_stage_ids=("reasoning", "final"),
        duration_ns=2_000_000,
        completed=False,
        error_type="RuntimeError",
    ))
    blocked_report = stats.report()
    assert blocked_report.stage_blocked_counts == {
        "reasoning": 1,
        "final": 1,
    }
    stats.record(SuperAIRunMetric(
        route=SuperAIRoute.FAST,
        signals=(),
        expected_stage_count=1,
        executed_stage_ids=("final",),
        failed_stage_ids=(),
        blocked_stage_ids=(),
        duration_ns=1_000_000,
        completed=True,
    ))

    assert len(stats.records()) == 1
    assert stats.report().route_counts == {"fast": 1}
    stats.clear()
    assert stats.report().total_runs == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: InMemorySuperAIStats(0),
        lambda: InMemorySuperAIStats().record(object()),
        lambda: SuperAIRunMetric(
            "fast", (), 1, (), (), (), 0, False
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            ("invalid",),
            1,
            (),
            (),
            (),
            0,
            False,
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (
                RoutingSignal.LONG_REQUEST,
                RoutingSignal.LONG_REQUEST,
            ),
            1,
            (),
            (),
            (),
            0,
            False,
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST, (), 0, (), (), (), 0, False
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            1,
            ("final",),
            ("other",),
            (),
            0,
            False,
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            1,
            ("final",),
            (),
            ("final",),
            0,
            False,
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            1,
            ("first",),
            (),
            ("second",),
            0,
            False,
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            1,
            (),
            (),
            (),
            -1,
            False,
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            1,
            (),
            (),
            (),
            0,
            "yes",
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            1,
            (),
            (),
            (),
            0,
            False,
            "invalid error",
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            1,
            "final",
            (),
            (),
            0,
            False,
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            1,
            ("invalid id",),
            (),
            (),
            0,
            False,
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            2,
            ("final", "final"),
            (),
            (),
            0,
            False,
        ),
        lambda: SuperAIRunMetric(
            SuperAIRoute.FAST,
            (),
            1,
            ("final",),
            (),
            (),
            0,
            True,
            "RuntimeError",
        ),
    ],
)
def test_stats_contract_rejects_invalid_data(factory):
    with pytest.raises(SuperAIStatsValidationError):
        factory()


def test_routed_client_validates_stats_dependencies():
    client = routed_client(StatsClient(), clock_ns=lambda: 1)

    with pytest.raises(TypeError, match="stats collector"):
        RoutedSuperAIClient(
            CapabilityRouter(),
            client.workflows,
            stats=object(),
        )
    with pytest.raises(TypeError, match="stats clock"):
        RoutedSuperAIClient(
            CapabilityRouter(),
            client.workflows,
            clock_ns=object(),
        )
