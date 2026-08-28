from collections.abc import Iterator

from ai_sdk.agents.handoff import (
    HandoffResult,
    SequentialHandoffCoordinator,
)
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.types import LLMMessage


class SuperAIClient(BaseLLMClient):
    """Expose a multi-provider handoff workflow as one LLM client."""

    def __init__(
        self,
        workflow: SequentialHandoffCoordinator,
    ) -> None:
        if not isinstance(
            workflow,
            SequentialHandoffCoordinator,
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
