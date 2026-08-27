import re
from collections.abc import Sequence
from typing import Protocol

from ai_sdk.llm.types import LLMMessage


class TokenCounter(Protocol):
    """Provider-neutral contract for estimating prompt tokens."""

    def count(self, text: str) -> int: ...


class RegexTokenCounter:
    """Deterministically approximate tokens without a model dependency."""

    def count(self, text: str) -> int:
        return len(re.findall(
            r"\w+|[^\w\s]",
            text,
        ))


class SlidingContextWindow:
    """Keep the newest complete conversation turns within a token budget."""

    def __init__(
        self,
        max_tokens: int,
        token_counter: TokenCounter | None = None,
        message_overhead: int = 4,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(
                "Context token budget must be greater than zero."
            )

        if message_overhead < 0:
            raise ValueError(
                "Message token overhead cannot be negative."
            )

        self.max_tokens = max_tokens
        self.token_counter = (
            token_counter or RegexTokenCounter()
        )
        self.message_overhead = message_overhead

    def select(
        self,
        messages: Sequence[LLMMessage],
    ) -> list[LLMMessage]:
        turns = self._group_turns(messages)

        if not turns:
            return []

        selected_turns: list[list[LLMMessage]] = []
        used_tokens = 0

        for turn in reversed(turns):
            turn_tokens = sum(
                self.token_counter.count(
                    message["content"]
                )
                + self.message_overhead
                for message in turn
            )

            if (
                selected_turns
                and used_tokens + turn_tokens
                > self.max_tokens
            ):
                break

            selected_turns.append(turn)
            used_tokens += turn_tokens

        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for turn in reversed(selected_turns)
            for message in turn
        ]

    @staticmethod
    def _group_turns(
        messages: Sequence[LLMMessage],
    ) -> list[list[LLMMessage]]:
        turns: list[list[LLMMessage]] = []

        for message in messages:
            copied_message: LLMMessage = {
                "role": message["role"],
                "content": message["content"],
            }

            if message["role"] == "user" or not turns:
                turns.append([copied_message])
            else:
                turns[-1].append(copied_message)

        return turns
