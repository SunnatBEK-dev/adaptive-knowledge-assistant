from collections.abc import Iterator

from anthropic import Anthropic

from ai_sdk.config import (
    API_KEY,
    MODEL,
    MAX_TOKENS,
    TIMEOUT,
)
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.types import LLMMessage


class ClaudeClient(BaseLLMClient):

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
