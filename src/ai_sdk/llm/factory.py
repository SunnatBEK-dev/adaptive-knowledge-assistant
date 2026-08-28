from ai_sdk.config import AI_PROVIDER
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.claude import ClaudeClient
from ai_sdk.llm.openai import OpenAIClient


def create_llm_client(
    provider: str | None = None,
) -> BaseToolLLMClient:
    """Create the configured provider adapter."""
    selected = AI_PROVIDER if provider is None else provider
    if not isinstance(selected, str) or not selected.strip():
        raise RuntimeError("AI_PROVIDER is not configured.")

    normalized = selected.strip().casefold()
    if normalized == "anthropic":
        return ClaudeClient()
    if normalized == "openai":
        return OpenAIClient()
    raise RuntimeError(
        f"Unsupported AI_PROVIDER: {normalized}."
    )
