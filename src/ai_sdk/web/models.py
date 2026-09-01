from typing import Literal

from pydantic import BaseModel, Field, model_validator

ProviderName = Literal["anthropic", "openai", "gemini"]
ChatMode = Literal["single", "adaptive"]


class ChatStreamRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    mode: ChatMode
    provider: ProviderName | None = None

    @model_validator(mode="after")
    def validate_provider(self) -> "ChatStreamRequest":
        self.message = self.message.strip()
        if not self.message:
            raise ValueError("Message cannot be blank.")
        if self.mode == "single" and self.provider is None:
            raise ValueError("Single Model mode requires a provider.")
        if self.mode == "adaptive" and self.provider is not None:
            raise ValueError(
                "Adaptive Multi-Model mode selects providers automatically."
            )
        return self


class ConversationResetRequest(BaseModel):
    mode: ChatMode
    provider: ProviderName | None = None

    @model_validator(mode="after")
    def validate_provider(self) -> "ConversationResetRequest":
        if self.mode == "single" and self.provider is None:
            raise ValueError("Single Model mode requires a provider.")
        if self.mode == "adaptive" and self.provider is not None:
            raise ValueError("Adaptive Multi-Model mode does not accept a provider.")
        return self


class ProviderReadinessResponse(BaseModel):
    provider: str
    display_name: str
    ready: bool
    missing_variables: list[str]


class AssistantStatusResponse(BaseModel):
    name: str
    version: str
    adaptive_ready: bool
    providers: list[ProviderReadinessResponse]
    document_count: int
    active_run_id: str | None
    adaptive_metrics: dict[str, object]


class IndexedDocumentResponse(BaseModel):
    document_id: str
    source: str
    format: str
    chunk_count: int
    page_count: int | None = None


class DocumentUploadResponse(BaseModel):
    document: IndexedDocumentResponse


class RunCancellationResponse(BaseModel):
    run_id: str
    accepted: bool


class ConversationResetResponse(BaseModel):
    reset: bool
