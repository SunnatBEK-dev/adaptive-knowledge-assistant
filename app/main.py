from ai_sdk.application.conversation_manager import (
    ConversationManager,
)
from ai_sdk.config import CHAT_FILE
from ai_sdk.context.prompt_builder import (
    PromptBuilder,
)
from ai_sdk.llm.claude import ClaudeClient
from ai_sdk.storage.json import (
    JsonConversationRepository,
)


def print_history(conversation) -> None:
    if conversation.is_empty():
        print("\nHistory is empty.\n")
        return

    print()

    for message in conversation.history():
        print(
            f"{message.role.capitalize()}: "
            f"{message.content}"
        )
        print(f"ID: {message.id}")
        print()


def main() -> None:
    repository = JsonConversationRepository(
        CHAT_FILE
    )

    conversation = repository.load()

    prompt_builder = PromptBuilder(
        conversation
    )

    client = ClaudeClient()

    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=prompt_builder,
        client=client,
        repository=repository,
    )

    print("Claude Chat")
    print(
        "Commands: "
        "/exit /save /clear /history\n"
    )

    while True:
        try:
            prompt = input("You: ").strip()

            if not prompt:
                continue

            command = prompt.lower()

            if command == "/exit":
                break

            if command == "/save":
                repository.save(conversation)
                print(
                    "Conversation saved.\n"
                )
                continue

            if command == "/clear":
                conversation.clear()
                repository.save(conversation)
                print(
                    "Conversation cleared.\n"
                )
                continue

            if command == "/history":
                print_history(conversation)
                continue

            print(
                "\nClaude: ",
                end="",
                flush=True,
            )

            for chunk in manager.stream_message(
                prompt
            ):
                print(
                    chunk,
                    end="",
                    flush=True,
                )

            print("\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except Exception as e:
            print(
                f"\nError: {e}\n"
            )


if __name__ == "__main__":
    main()
