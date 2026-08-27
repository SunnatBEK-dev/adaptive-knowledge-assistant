from abc import ABC, abstractmethod
from collections.abc import Iterator

from ai_sdk.llm.types import LLMMessage


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
