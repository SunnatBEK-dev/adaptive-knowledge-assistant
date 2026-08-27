from abc import ABC, abstractmethod
from collections.abc import Iterator

from ai_sdk.llm.types import LLMMessage
from ai_sdk.tools.executor import ToolExecutor


class BaseLLMClient(ABC):

    @abstractmethod
    def ask(
        self,
        messages: list[LLMMessage],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
    ) -> Iterator[str]:
        raise NotImplementedError


class BaseToolLLMClient(BaseLLMClient):
    """LLM adapter that can complete a bounded tool-use loop."""

    @abstractmethod
    def ask_with_tools(
        self,
        messages: list[LLMMessage],
        executor: ToolExecutor,
        *,
        max_tool_rounds: int = 8,
    ) -> str:
        raise NotImplementedError
