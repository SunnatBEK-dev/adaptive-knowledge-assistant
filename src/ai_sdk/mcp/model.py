from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import re
from urllib.parse import urlsplit


MCP_PROTOCOL_VERSION = "2026-07-28"

PROTOCOL_VERSION_META_KEY = (
    "io.modelcontextprotocol/protocolVersion"
)
CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"
CLIENT_CAPABILITIES_META_KEY = (
    "io.modelcontextprotocol/clientCapabilities"
)

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class MCPValidationError(ValueError):
    """Raised when a local MCP contract is invalid."""


class MCPConnectionState(str, Enum):
    NEW = "new"
    OPENING = "opening"
    OPEN = "open"
    FAILED = "failed"
    CLOSED = "closed"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MCPValidationError(
            f"MCP {field_name} cannot be empty."
        )
    return value


def _copy_mapping(
    value: object,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MCPValidationError(
            f"MCP {field_name} must be an object."
        )
    if any(not isinstance(key, str) for key in value):
        raise MCPValidationError(
            f"MCP {field_name} keys must be strings."
        )
    return deepcopy(dict(value))


def _optional_text(
    value: object,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _validate_cache_hints(
    ttl_ms: object,
    cache_scope: object,
) -> tuple[int | None, str | None]:
    if ttl_ms is not None and (
        not isinstance(ttl_ms, int)
        or isinstance(ttl_ms, bool)
        or ttl_ms < 0
    ):
        raise MCPValidationError(
            "MCP cache TTL must be a non-negative integer."
        )
    return ttl_ms, _optional_text(cache_scope, "cache scope")


@dataclass(frozen=True)
class MCPImplementation:
    name: str
    version: str

    def __post_init__(self) -> None:
        _require_text(self.name, "implementation name")
        _require_text(self.version, "implementation version")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, init=False)
class MCPRequestContext:
    protocol_version: str
    client_info: MCPImplementation
    client_capabilities: dict[str, object]

    def __init__(
        self,
        protocol_version: str,
        client_info: MCPImplementation,
        client_capabilities: Mapping[str, object] | None = None,
    ) -> None:
        _require_text(protocol_version, "protocol version")
        if not isinstance(client_info, MCPImplementation):
            raise MCPValidationError(
                "MCP client info is invalid."
            )

        capabilities = _copy_mapping(
            {}
            if client_capabilities is None
            else client_capabilities,
            "client capabilities",
        )
        object.__setattr__(
            self,
            "protocol_version",
            protocol_version,
        )
        object.__setattr__(self, "client_info", client_info)
        object.__setattr__(
            self,
            "client_capabilities",
            capabilities,
        )

    def to_meta(self) -> dict[str, object]:
        return {
            PROTOCOL_VERSION_META_KEY: self.protocol_version,
            CLIENT_INFO_META_KEY: self.client_info.to_dict(),
            CLIENT_CAPABILITIES_META_KEY: deepcopy(
                self.client_capabilities
            ),
        }


@dataclass(frozen=True, init=False)
class MCPServerCapabilities:
    tools: dict[str, object] | None
    resources: dict[str, object] | None

    def __init__(
        self,
        *,
        tools: Mapping[str, object] | None = None,
        resources: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "tools",
            None
            if tools is None
            else _copy_mapping(tools, "tools capability"),
        )
        object.__setattr__(
            self,
            "resources",
            None
            if resources is None
            else _copy_mapping(
                resources,
                "resources capability",
            ),
        )

    @property
    def supports_tools(self) -> bool:
        return self.tools is not None

    @property
    def supports_resources(self) -> bool:
        return self.resources is not None


@dataclass(frozen=True, init=False)
class MCPDiscoveryResult:
    supported_versions: tuple[str, ...]
    capabilities: MCPServerCapabilities
    server_info: MCPImplementation | None
    instructions: str | None
    ttl_ms: int | None
    cache_scope: str | None

    def __init__(
        self,
        supported_versions: Sequence[str],
        capabilities: MCPServerCapabilities,
        *,
        server_info: MCPImplementation | None = None,
        instructions: str | None = None,
        ttl_ms: int | None = None,
        cache_scope: str | None = None,
    ) -> None:
        if isinstance(supported_versions, (str, bytes)):
            raise MCPValidationError(
                "MCP supported versions must be a sequence."
            )
        versions = tuple(supported_versions)
        if not versions or any(
            not isinstance(version, str) or not version.strip()
            for version in versions
        ):
            raise MCPValidationError(
                "MCP supported versions must contain non-empty strings."
            )
        if len(set(versions)) != len(versions):
            raise MCPValidationError(
                "MCP supported versions must be unique."
            )
        if not isinstance(capabilities, MCPServerCapabilities):
            raise MCPValidationError(
                "MCP server capabilities are invalid."
            )
        if server_info is not None and not isinstance(
            server_info,
            MCPImplementation,
        ):
            raise MCPValidationError(
                "MCP server info is invalid."
            )
        validated_ttl, validated_scope = _validate_cache_hints(
            ttl_ms,
            cache_scope,
        )

        object.__setattr__(self, "supported_versions", versions)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "server_info", server_info)
        object.__setattr__(
            self,
            "instructions",
            _optional_text(instructions, "server instructions"),
        )
        object.__setattr__(self, "ttl_ms", validated_ttl)
        object.__setattr__(self, "cache_scope", validated_scope)


@dataclass(frozen=True, init=False)
class MCPTool:
    name: str
    input_schema: dict[str, object]
    title: str | None
    description: str | None

    def __init__(
        self,
        name: str,
        input_schema: Mapping[str, object],
        *,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        if (
            not isinstance(name, str)
            or _TOOL_NAME_PATTERN.fullmatch(name) is None
        ):
            raise MCPValidationError(
                "MCP tool name must use 1-128 ASCII letters, "
                "digits, underscores, hyphens, or dots."
            )
        schema = _copy_mapping(input_schema, "tool input schema")
        if schema.get("type") != "object":
            raise MCPValidationError(
                "MCP tool input schema root type must be object."
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "input_schema", schema)
        object.__setattr__(
            self,
            "title",
            _optional_text(title, "tool title"),
        )
        object.__setattr__(
            self,
            "description",
            _optional_text(description, "tool description"),
        )


@dataclass(frozen=True, init=False)
class MCPResource:
    uri: str
    name: str
    title: str | None
    description: str | None
    mime_type: str | None
    size: int | None

    def __init__(
        self,
        uri: str,
        name: str,
        *,
        title: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        size: int | None = None,
    ) -> None:
        validated_uri = _require_text(uri, "resource URI")
        try:
            uri_scheme = urlsplit(validated_uri).scheme
        except ValueError as error:
            raise MCPValidationError(
                "MCP resource URI is invalid."
            ) from error
        if not uri_scheme:
            raise MCPValidationError(
                "MCP resource URI must include a scheme."
            )
        if size is not None and (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise MCPValidationError(
                "MCP resource size must be a non-negative integer."
            )

        object.__setattr__(self, "uri", validated_uri)
        object.__setattr__(self, "name", _require_text(name, "resource name"))
        object.__setattr__(
            self,
            "title",
            _optional_text(title, "resource title"),
        )
        object.__setattr__(
            self,
            "description",
            _optional_text(description, "resource description"),
        )
        object.__setattr__(
            self,
            "mime_type",
            _optional_text(mime_type, "resource MIME type"),
        )
        object.__setattr__(self, "size", size)


@dataclass(frozen=True, init=False)
class MCPToolPage:
    tools: tuple[MCPTool, ...]
    next_cursor: str | None
    ttl_ms: int | None
    cache_scope: str | None

    def __init__(
        self,
        tools: Sequence[MCPTool],
        *,
        next_cursor: str | None = None,
        ttl_ms: int | None = None,
        cache_scope: str | None = None,
    ) -> None:
        normalized = tuple(tools)
        if any(not isinstance(tool, MCPTool) for tool in normalized):
            raise MCPValidationError(
                "MCP tool page contains an invalid tool."
            )
        validated_ttl, validated_scope = _validate_cache_hints(
            ttl_ms,
            cache_scope,
        )
        object.__setattr__(self, "tools", normalized)
        object.__setattr__(
            self,
            "next_cursor",
            _optional_text(next_cursor, "next cursor"),
        )
        object.__setattr__(self, "ttl_ms", validated_ttl)
        object.__setattr__(self, "cache_scope", validated_scope)


@dataclass(frozen=True, init=False)
class MCPResourcePage:
    resources: tuple[MCPResource, ...]
    next_cursor: str | None
    ttl_ms: int | None
    cache_scope: str | None

    def __init__(
        self,
        resources: Sequence[MCPResource],
        *,
        next_cursor: str | None = None,
        ttl_ms: int | None = None,
        cache_scope: str | None = None,
    ) -> None:
        normalized = tuple(resources)
        if any(
            not isinstance(resource, MCPResource)
            for resource in normalized
        ):
            raise MCPValidationError(
                "MCP resource page contains an invalid resource."
            )
        validated_ttl, validated_scope = _validate_cache_hints(
            ttl_ms,
            cache_scope,
        )
        object.__setattr__(self, "resources", normalized)
        object.__setattr__(
            self,
            "next_cursor",
            _optional_text(next_cursor, "next cursor"),
        )
        object.__setattr__(self, "ttl_ms", validated_ttl)
        object.__setattr__(self, "cache_scope", validated_scope)
