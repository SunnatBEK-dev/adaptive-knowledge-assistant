from collections.abc import Iterator

from ai_sdk.core.conversation import Conversation
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.types import LLMMessage
from ai_sdk.storage.base import ConversationRepository


class ConversationManager:

    def __init__(
        self,
        conversation: Conversation,
        prompt_builder: PromptBuilder,
        client: BaseLLMClient,
        repository: ConversationRepository,
    ) -> None:
        self.conversation = conversation
        self.prompt_builder = prompt_builder
        self.client = client
        self.repository = repository

    def _build_messages(
        self,
        text: str,
    ) -> list[LLMMessage]:
        return self.prompt_builder.build_messages()

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
