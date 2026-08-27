import copy

from ai_sdk.core.conversation import Conversation
from ai_sdk.llm.types import LLMMessage


class PromptBuilder:

    def __init__(
        self,
        conversation: Conversation,
    ) -> None:
        self.conversation = conversation

    def build_messages(
        self,
    ) -> list[LLMMessage]:
        messages: list[LLMMessage] = []

        history = copy.deepcopy(
            self.conversation.history()
        )

        for message in history:
            messages.append({
                "role": message.role,
                "content": message.content,
            })

        return messages
