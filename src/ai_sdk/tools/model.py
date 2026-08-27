from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]

    def __init__(
        self,
        id: str,
        name: str,
        arguments: Mapping[str, object],
    ) -> None:
        if not isinstance(id, str) or not id.strip():
            raise ValueError("Tool call ID cannot be empty.")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool call name cannot be empty.")

        if not isinstance(arguments, Mapping):
            raise ValueError(
                "Tool call arguments must be an object."
            )

        object.__setattr__(self, "id", id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments", dict(arguments))


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False
