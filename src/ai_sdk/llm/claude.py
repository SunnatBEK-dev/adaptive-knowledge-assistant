from collections.abc import Iterator

from anthropic import Anthropic

from ai_sdk.config import (
    API_KEY,
    MODEL,
    MAX_TOKENS,
    TIMEOUT,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.types import LLMMessage
from ai_sdk.tools.executor import ToolExecutor
from ai_sdk.tools.model import ToolCall


class ClaudeClient(BaseToolLLMClient):

    def __init__(
        self,
        client: Anthropic | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = MAX_TOKENS,
        timeout: float = TIMEOUT,
    ) -> None:
        self.model = model or MODEL
        self.max_tokens = max_tokens

        if not self.model:
            raise RuntimeError(
                "MODEL is not configured."
            )

        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or API_KEY

        if not resolved_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not configured."
            )

        self.client = Anthropic(
            api_key=resolved_api_key,
            timeout=timeout,
        )

    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )

        parts = []

        for block in response.content:
            if block.type == "text":
                parts.append(block.text)

        return "".join(parts)

    def ask_with_tools(
        self,
        messages: list[LLMMessage],
        executor: ToolExecutor,
        *,
        max_tool_rounds: int = 8,
    ) -> str:
        """Ask Claude and execute requested tools until final text."""
        if (
            not isinstance(max_tool_rounds, int)
            or isinstance(max_tool_rounds, bool)
            or max_tool_rounds <= 0
        ):
            raise ValueError(
                "Maximum tool rounds must be greater than zero."
            )

        if not isinstance(executor, ToolExecutor):
            raise TypeError(
                "Tool executor must be a ToolExecutor."
            )

        schemas = executor.registry.provider_schemas()

        if not schemas:
            return self.ask(messages)

        provider_messages: list[dict[str, object]] = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
        ]
        completed_tool_rounds = 0
        seen_call_ids: set[str] = set()

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=provider_messages,
                tools=schemas,
            )
            text_parts, calls, assistant_blocks = (
                self._parse_tool_response(response.content)
            )

            if not calls:
                if getattr(response, "stop_reason", None) == "tool_use":
                    raise RuntimeError(
                        "Claude returned tool_use without a tool call."
                    )

                return "".join(text_parts)

            if completed_tool_rounds >= max_tool_rounds:
                raise RuntimeError(
                    "Maximum Claude tool rounds exceeded."
                )

            call_ids = [call.id for call in calls]
            duplicate_ids = (
                len(call_ids) != len(set(call_ids))
                or any(
                    call_id in seen_call_ids
                    for call_id in call_ids
                )
            )

            if duplicate_ids:
                raise RuntimeError(
                    "Claude returned a duplicate tool call ID."
                )

            seen_call_ids.update(call_ids)
            provider_messages.append({
                "role": "assistant",
                "content": assistant_blocks,
            })
            results = [
                executor.execute(call)
                for call in calls
            ]
            provider_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                    for result in results
                ],
            })
            completed_tool_rounds += 1

    @staticmethod
    def _parse_tool_response(
        blocks: object,
    ) -> tuple[
        list[str],
        list[ToolCall],
        list[dict[str, object]],
    ]:
        if not isinstance(blocks, list):
            raise RuntimeError(
                "Claude response content must be a list."
            )

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        assistant_blocks: list[dict[str, object]] = []

        for block in blocks:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text = getattr(block, "text", None)

                if not isinstance(text, str):
                    raise RuntimeError(
                        "Claude text block is invalid."
                    )

                text_parts.append(text)
                assistant_blocks.append({
                    "type": "text",
                    "text": text,
                })
                continue

            if block_type == "tool_use":
                try:
                    call = ToolCall(
                        id=getattr(block, "id", None),
                        name=getattr(block, "name", None),
                        arguments=getattr(block, "input", None),
                    )
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        "Claude tool-use block is invalid."
                    ) from error

                calls.append(call)
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                })

        return text_parts, calls, assistant_blocks

    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        ) as stream:

            for text in stream.text_stream:
                yield text
