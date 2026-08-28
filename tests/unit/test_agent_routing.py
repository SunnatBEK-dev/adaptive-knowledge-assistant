import pytest

from ai_sdk.agents import (
    AICapability,
    AgentModelResponse,
    AgentRunner,
    AgentTextBlock,
    AgentWorker,
    CapabilityRouter,
    CoordinationError,
    HandoffStage,
    MultiAgentCoordinator,
    RoutingDecision,
    RoutingSignal,
    SequentialHandoffCoordinator,
    SuperAIRoute,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.super_ai import (
    RoutedSuperAIClient,
    SuperAIClient,
)
from ai_sdk.tools import ToolExecutor, ToolRegistry


@pytest.mark.parametrize(
    ("text", "expected_route"),
    [
        ("Salom", SuperAIRoute.FAST),
        (
            "Ushbu hujjatdagi manbalarni ko'rsat",
            SuperAIRoute.CONTEXT,
        ),
        ("Nega bu yechim ishlaydi?", SuperAIRoute.REASONING),
        (
            "Manbalarni chuqur tahlil qilib taqqosla",
            SuperAIRoute.FULL,
        ),
        (
            "Retrieved context:\nPython facts",
            SuperAIRoute.CONTEXT,
        ),
        (
            "Birinchi savol? Ikkinchi savol?",
            SuperAIRoute.REASONING,
        ),
        (
            "1. Birinchi vazifa\n2. Ikkinchi vazifa",
            SuperAIRoute.REASONING,
        ),
        ("Natijani tekshirib ber", SuperAIRoute.CONTEXT),
        ("Loyihani tahlilini ber", SuperAIRoute.REASONING),
    ],
)
def test_capability_router_selects_explainable_route(
    text,
    expected_route,
):
    decision = CapabilityRouter().route(text)

    assert decision.route is expected_route
    assert decision.capabilities[-1] is AICapability.SYNTHESIS


def test_capability_router_marks_long_request_as_full():
    decision = CapabilityRouter(
        long_request_chars=10,
        max_analyzed_chars=20,
    ).route("x" * 30)

    assert decision.route is SuperAIRoute.FULL
    assert decision.signals == (RoutingSignal.LONG_REQUEST,)
    assert decision.capabilities == (
        AICapability.CONTEXT,
        AICapability.REASONING,
        AICapability.SYNTHESIS,
    )


def test_routing_decision_and_router_validate_configuration():
    with pytest.raises(TypeError, match="route"):
        RoutingDecision("fast")
    with pytest.raises(TypeError, match="signals"):
        RoutingDecision(SuperAIRoute.FAST, ("invalid",))
    with pytest.raises(CoordinationError, match="unique"):
        RoutingDecision(
            SuperAIRoute.FAST,
            (
                RoutingSignal.LONG_REQUEST,
                RoutingSignal.LONG_REQUEST,
            ),
        )
    with pytest.raises(ValueError, match="positive"):
        CapabilityRouter(long_request_chars=0)
    with pytest.raises(ValueError, match="cannot be below"):
        CapabilityRouter(
            long_request_chars=10,
            max_analyzed_chars=5,
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        CapabilityRouter().route(" ")


class RouteClient(BaseToolLLMClient):
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def ask(self, messages):
        return self.response

    def stream(self, messages):
        yield self.response

    def complete_tool_turn(self, messages, schemas, events):
        self.prompts.append(messages[0]["content"])
        return AgentModelResponse([
            AgentTextBlock(self.response),
        ])


def workflow(route):
    provider = RouteClient(route.value)
    worker = AgentWorker(
        f"{route.value}_worker",
        "Return route response",
        AgentRunner(
            provider,
            ToolExecutor(ToolRegistry()),
        ),
    )
    client = SuperAIClient(SequentialHandoffCoordinator(
        MultiAgentCoordinator([worker]),
        [HandoffStage(
            "final",
            worker.name,
            "Answer",
        )],
    ))
    return client, provider


def routed_client():
    clients = {}
    providers = {}
    for route in SuperAIRoute:
        client, provider = workflow(route)
        clients[route] = client
        providers[route] = provider
    return (
        RoutedSuperAIClient(CapabilityRouter(), clients),
        providers,
    )


@pytest.mark.parametrize(
    ("text", "expected_route"),
    [
        ("Salom", SuperAIRoute.FAST),
        ("Manbani ko'rsat", SuperAIRoute.CONTEXT),
        ("Nega shunday?", SuperAIRoute.REASONING),
        ("Manbani tahlil qil", SuperAIRoute.FULL),
    ],
)
def test_routed_super_ai_executes_only_selected_workflow(
    text,
    expected_route,
):
    client, providers = routed_client()

    response = client.ask([{
        "role": "user",
        "content": text,
    }])

    assert response == expected_route.value
    assert client.last_decision.route is expected_route
    assert client.last_result.completed is True
    assert providers[expected_route].prompts
    assert all(
        not provider.prompts
        for route, provider in providers.items()
        if route is not expected_route
    )


def test_routed_super_ai_validates_workflows_and_messages():
    client, _ = routed_client()
    valid_workflows = client.workflows

    with pytest.raises(TypeError, match="router"):
        RoutedSuperAIClient(object(), valid_workflows)
    with pytest.raises(TypeError, match="mapping"):
        RoutedSuperAIClient(CapabilityRouter(), [])
    invalid_keys = dict(valid_workflows)
    fast_workflow = invalid_keys.pop(SuperAIRoute.FAST)
    invalid_keys["fast"] = fast_workflow
    with pytest.raises(TypeError, match="keys"):
        RoutedSuperAIClient(CapabilityRouter(), invalid_keys)
    with pytest.raises(ValueError, match="every route"):
        RoutedSuperAIClient(
            CapabilityRouter(),
            {SuperAIRoute.FAST: valid_workflows[SuperAIRoute.FAST]},
        )
    invalid_workflows = dict(valid_workflows)
    invalid_workflows[SuperAIRoute.FAST] = object()
    with pytest.raises(TypeError, match="SuperAIClient"):
        RoutedSuperAIClient(
            CapabilityRouter(),
            invalid_workflows,
        )
    with pytest.raises(RuntimeError, match="streaming"):
        client.stream([{"role": "user", "content": "Hi"}])
    with pytest.raises(ValueError, match="cannot be empty"):
        client.ask([])
    with pytest.raises(ValueError, match="requires a user"):
        client.ask([{"role": "assistant", "content": "Hi"}])
    with pytest.raises(TypeError, match="content"):
        client.ask([{"role": "user", "content": None}])
