from enum import Enum


class ApplicationMode(str, Enum):
    """User-visible application sections."""

    DIRECT_CHAT = "direct_chat"
    SUPER_AI = "super_ai"
