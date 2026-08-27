import json

from ai_sdk.tools.model import ToolCall, ToolResult
from ai_sdk.tools.registry import ToolRegistry
from ai_sdk.tools.schema import ToolValidationError


class ToolExecutor:
    """Validate and execute only handlers in an explicit registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self.registry.get(call.name)

        if tool is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"Unknown tool: {call.name}",
                is_error=True,
            )

        try:
            arguments = tool.schema.validate_arguments(
                call.arguments
            )
            output = tool.handler(**arguments)
            content = self._serialize_output(output)
        except ToolValidationError as error:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=str(error),
                is_error=True,
            )
        except Exception as error:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=(
                    "Tool execution failed: "
                    f"{type(error).__name__}"
                ),
                is_error=True,
            )

        return ToolResult(
            call_id=call.id,
            name=call.name,
            content=content,
        )

    @staticmethod
    def _serialize_output(output: object) -> str:
        if isinstance(output, str):
            return output

        return json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
