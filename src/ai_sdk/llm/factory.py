from ai_sdk.config import AI_PROVIDER
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.llm.claude import ClaudeClient
from ai_sdk.llm.gemini import GeminiClient
from ai_sdk.llm.openai import OpenAIClient


SUPPORTED_LLM_PROVIDERS = (
    "anthropic",
    "openai",
    "gemini",
)


def normalize_llm_provider(provider: object) -> str:
    """Validate and normalize a built-in provider name."""
    if not isinstance(provider, str) or not provider.strip():
        raise RuntimeError("AI provider is not configured.")

    normalized = provider.strip().casefold()
    if normalized not in SUPPORTED_LLM_PROVIDERS:
        raise RuntimeError(
            f"Unsupported AI provider: {normalized}."
        )
    return normalized


def create_llm_client(
    provider: str | None = None,
) -> BaseToolLLMClient:
    """Create the configured provider adapter."""
    selected = AI_PROVIDER if provider is None else provider
    normalized = normalize_llm_provider(selected)
    factories = {
        "anthropic": ClaudeClient,
        "openai": OpenAIClient,
        "gemini": GeminiClient,
    }
    return factories[normalized]()
