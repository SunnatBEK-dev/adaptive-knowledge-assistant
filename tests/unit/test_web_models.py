import pytest
from pydantic import ValidationError

from ai_sdk.web.models import ChatStreamRequest, ConversationResetRequest


def test_single_model_chat_requires_provider_and_strips_message():
    request = ChatStreamRequest(
        message="  Explain retrieval  ",
        mode="single",
        provider="openai",
    )

    assert request.message == "Explain retrieval"
    assert request.provider == "openai"

    with pytest.raises(ValidationError, match="requires a provider"):
        ChatStreamRequest(message="question", mode="single")


def test_adaptive_request_rejects_explicit_provider_and_blank_message():
    with pytest.raises(ValidationError, match="automatically"):
        ChatStreamRequest(
            message="question",
            mode="adaptive",
            provider="openai",
        )

    with pytest.raises(ValidationError, match="blank"):
        ChatStreamRequest(message="   ", mode="adaptive")


def test_conversation_reset_validates_mode_provider_pair():
    assert (
        ConversationResetRequest(
            mode="single",
            provider="gemini",
        ).provider
        == "gemini"
    )
    assert ConversationResetRequest(mode="adaptive").provider is None

    with pytest.raises(ValidationError, match="requires a provider"):
        ConversationResetRequest(mode="single")
    with pytest.raises(ValidationError, match="does not accept"):
        ConversationResetRequest(mode="adaptive", provider="gemini")
