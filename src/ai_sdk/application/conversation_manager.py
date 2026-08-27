from collections.abc import Iterator

from ai_sdk.core.conversation import Conversation
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.types import LLMMessage
from ai_sdk.memory.base import BaseMemoryStore
from ai_sdk.memory.model import (
    LongTermMemory,
    MemorySearchResult,
)
from ai_sdk.storage.base import ConversationRepository


class ConversationManager:

    def __init__(
        self,
        conversation: Conversation,
        prompt_builder: PromptBuilder,
        client: BaseLLMClient,
        repository: ConversationRepository,
        memory_store: BaseMemoryStore | None = None,
        memory_retrieval_k: int = 3,
    ) -> None:
        if memory_retrieval_k <= 0:
            raise ValueError(
                "Memory retrieval top-k must be greater than zero."
            )

        self.conversation = conversation
        self.prompt_builder = prompt_builder
        self.client = client
        self.repository = repository
        self.memory_store = memory_store
        self.memory_retrieval_k = memory_retrieval_k

    def _build_messages(
        self,
        text: str,
    ) -> list[LLMMessage]:
        return self.prompt_builder.build_messages(
            memory_results=self._recall_memories(text)
        )

    def remember(self, content: str) -> LongTermMemory:
        memory_store = self._require_memory_store()
        normalized_content = content.strip()

        if not normalized_content:
            raise ValueError(
                "Long-term memory content cannot be empty."
            )

        for memory in memory_store.list_memories():
            if (
                memory.content.casefold()
                == normalized_content.casefold()
            ):
                return memory

        memory = LongTermMemory.create(normalized_content)
        memory_store.add(memory)
        return memory

    def list_memories(self) -> list[LongTermMemory]:
        return self._require_memory_store().list_memories()

    def forget(self, memory_id: str) -> bool:
        return self._require_memory_store().delete(memory_id)

    def _recall_memories(
        self,
        query: str,
    ) -> list[MemorySearchResult]:
        if self.memory_store is None or not query.strip():
            return []

        return self.memory_store.search(
            query=query,
            k=self.memory_retrieval_k,
        )

    def _require_memory_store(self) -> BaseMemoryStore:
        if self.memory_store is None:
            raise RuntimeError(
                "Long-term memory is not configured."
            )

        return self.memory_store

    def send_message(
        self,
        text: str,
    ) -> str:
        user_message = (
            self.conversation.add_user(text)
        )
        assistant_message = None

        try:
            messages = self._build_messages(text)

            response = self.client.ask(
                messages
            )

            assistant_message = (
                self.conversation.add_assistant(
                    response
                )
            )

            self.repository.save(
                self.conversation
            )

            return response

        except Exception:
            if assistant_message is not None:
                self.conversation.delete_message(
                    assistant_message.id
                )

            self.conversation.delete_message(
                user_message.id
            )

            raise

    def stream_message(
        self,
        text: str,
    ) -> Iterator[str]:
        user_message = (
            self.conversation.add_user(text)
        )

        chunks = []
        assistant_message = None

        try:
            messages = self._build_messages(text)

            for chunk in self.client.stream(
                messages
            ):
                chunks.append(chunk)
                yield chunk

            response = "".join(chunks)

            assistant_message = (
                self.conversation.add_assistant(
                    response
                )
            )

            self.repository.save(
                self.conversation
            )

        except (Exception, GeneratorExit):
            if assistant_message is not None:
                self.conversation.delete_message(
                    assistant_message.id
                )

            self.conversation.delete_message(
                user_message.id
            )

            raise
