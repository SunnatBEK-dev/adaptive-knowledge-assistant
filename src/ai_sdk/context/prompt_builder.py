from collections.abc import Sequence

from ai_sdk.core.conversation import Conversation
from ai_sdk.llm.types import LLMMessage
from ai_sdk.retrieval.search import SearchResult


class PromptBuilder:

    def __init__(
        self,
        conversation: Conversation,
    ) -> None:
        self.conversation = conversation

    def build_messages(
        self,
        retrieval_results: Sequence[
            SearchResult
        ] = (),
    ) -> list[LLMMessage]:
        messages: list[LLMMessage] = []

        for message in self.conversation.history():
            messages.append({
                "role": message.role,
                "content": message.content,
            })

        if retrieval_results:
            self._augment_latest_user_message(
                messages,
                retrieval_results,
            )

        return messages

    @staticmethod
    def _augment_latest_user_message(
        messages: list[LLMMessage],
        retrieval_results: Sequence[
            SearchResult
        ],
    ) -> None:
        for message in reversed(messages):
            if message["role"] != "user":
                continue

            context = "\n\n".join(
                f"[{index}]\n{result.chunk.content}"
                for index, result in enumerate(
                    retrieval_results,
                    start=1,
                )
            )
            question = message["content"]
            message["content"] = (
                "Retrieved context:\n"
                f"{context}\n\n"
                "User question:\n"
                f"{question}"
            )
            return

        raise RuntimeError(
            "Retrieval context requires a user message."
        )
