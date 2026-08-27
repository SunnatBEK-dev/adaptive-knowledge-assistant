from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation


def test_build_messages_preserves_order_without_internal_metadata():
    conversation = Conversation()
    conversation.add_user("Hello")
    conversation.add_assistant("Hi")

    messages = PromptBuilder(conversation).build_messages()

    assert messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    assert all("id" not in message for message in messages)


def test_build_messages_result_does_not_mutate_conversation():
    conversation = Conversation()
    original = conversation.add_user("Original")
    messages = PromptBuilder(conversation).build_messages()

    messages[0]["content"] = "Changed"

    assert original.content == "Original"
