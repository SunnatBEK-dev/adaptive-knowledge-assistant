from collections.abc import Callable, Mapping
from typing import TypeVar

from ai_sdk.mcp.model import (
    MCP_PROTOCOL_VERSION,
    MCPConnectionState,
    MCPDiscoveryResult,
    MCPImplementation,
    MCPRequestContext,
    MCPResourcePage,
    MCPToolPage,
    MCPValidationError,
)
from ai_sdk.mcp.transport import BaseMCPTransport


class MCPClientError(RuntimeError):
    """Base error for MCP client runtime failures."""


class MCPLifecycleError(MCPClientError):
    """Raised when an operation is invalid for the client state."""


class MCPTimeoutError(MCPClientError):
    """Raised when an MCP transport operation times out."""


class MCPTransportError(MCPClientError):
    """Raised when the transport fails without exposing secrets."""


class MCPProtocolError(MCPClientError):
    """Raised when a transport returns an invalid MCP result."""


class MCPCapabilityError(MCPClientError):
    """Raised when discovery says an operation is unsupported."""


ResultT = TypeVar("ResultT")


class MCPClient:
    """Stateless MCP 2026-07-28 client with transport lifecycle."""

    def __init__(
        self,
        transport: BaseMCPTransport,
        *,
        client_info: MCPImplementation,
        client_capabilities: Mapping[str, object] | None = None,
        protocol_version: str = MCP_PROTOCOL_VERSION,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not isinstance(transport, BaseMCPTransport):
            raise MCPValidationError("MCP transport is invalid.")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise MCPValidationError(
                "MCP timeout must be greater than zero."
            )

        self._transport = transport
        self._context = MCPRequestContext(
            protocol_version,
            client_info,
            client_capabilities,
        )
        self._timeout_seconds = float(timeout_seconds)
        self._state = MCPConnectionState.NEW
        self._transport_opened = False
        self._discovery: MCPDiscoveryResult | None = None

    @property
    def state(self) -> MCPConnectionState:
        return self._state

    @property
    def request_context(self) -> MCPRequestContext:
        return self._context

    @property
    def discovery(self) -> MCPDiscoveryResult | None:
        return self._discovery

    def open(self) -> None:
        if self._state is not MCPConnectionState.NEW:
            raise MCPLifecycleError(
                "MCP client can only open from the new state."
            )

        self._state = MCPConnectionState.OPENING
        try:
            self._transport.open(
                timeout_seconds=self._timeout_seconds
            )
        except TimeoutError as error:
            self._abort()
            raise MCPTimeoutError(
                "MCP transport open timed out."
            ) from error
        except Exception as error:
            self._abort()
            raise MCPTransportError(
                "MCP transport open failed: "
                f"{type(error).__name__}"
            ) from error

        self._transport_opened = True
        self._state = MCPConnectionState.OPEN

    def discover(self) -> MCPDiscoveryResult:
        result = self._request(
            "server discovery",
            lambda: self._transport.discover(
                self._context,
                timeout_seconds=self._timeout_seconds,
            ),
        )
        if not isinstance(result, MCPDiscoveryResult):
            self._fail_protocol(
                "MCP discovery returned an invalid result."
            )
        if (
            self._context.protocol_version
            not in result.supported_versions
        ):
            self._fail_protocol(
                "MCP server does not support the requested protocol "
                "version."
            )

        self._discovery = result
        return result

    def list_tools(
        self,
        *,
        cursor: str | None = None,
    ) -> MCPToolPage:
        self._require_open()
        self._validate_cursor(cursor)
        if (
            self._discovery is not None
            and not self._discovery.capabilities.supports_tools
        ):
            raise MCPCapabilityError(
                "MCP server did not advertise tools support."
            )

        page = self._request(
            "tools/list",
            lambda: self._transport.list_tools(
                self._context,
                cursor=cursor,
                timeout_seconds=self._timeout_seconds,
            ),
        )
        if not isinstance(page, MCPToolPage):
            self._fail_protocol(
                "MCP tools/list returned an invalid page."
            )
        names = [tool.name for tool in page.tools]
        if len(set(names)) != len(names):
            self._fail_protocol(
                "MCP tools/list returned duplicate tool names."
            )
        return page

    def list_resources(
        self,
        *,
        cursor: str | None = None,
    ) -> MCPResourcePage:
        self._require_open()
        self._validate_cursor(cursor)
        if (
            self._discovery is not None
            and not self._discovery.capabilities.supports_resources
        ):
            raise MCPCapabilityError(
                "MCP server did not advertise resources support."
            )

        page = self._request(
            "resources/list",
            lambda: self._transport.list_resources(
                self._context,
                cursor=cursor,
                timeout_seconds=self._timeout_seconds,
            ),
        )
        if not isinstance(page, MCPResourcePage):
            self._fail_protocol(
                "MCP resources/list returned an invalid page."
            )
        uris = [resource.uri for resource in page.resources]
        if len(set(uris)) != len(uris):
            self._fail_protocol(
                "MCP resources/list returned duplicate resource URIs."
            )
        return page

    def close(self) -> None:
        if self._state is MCPConnectionState.CLOSED:
            return

        should_close_transport = (
            self._transport_opened
            or self._state is MCPConnectionState.OPENING
        )
        self._transport_opened = False
        self._state = MCPConnectionState.CLOSED

        if not should_close_transport:
            return
        try:
            self._transport.close()
        except Exception as error:
            raise MCPTransportError(
                "MCP transport close failed: "
                f"{type(error).__name__}"
            ) from error

    def __enter__(self) -> "MCPClient":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _request(
        self,
        operation: str,
        callback: Callable[[], ResultT],
    ) -> ResultT:
        self._require_open()
        try:
            return callback()
        except TimeoutError as error:
            self._abort()
            raise MCPTimeoutError(
                f"MCP {operation} timed out."
            ) from error
        except Exception as error:
            self._abort()
            raise MCPTransportError(
                f"MCP {operation} failed: {type(error).__name__}"
            ) from error

    def _require_open(self) -> None:
        if self._state is not MCPConnectionState.OPEN:
            raise MCPLifecycleError(
                "MCP client operation requires an open transport."
            )

    @staticmethod
    def _validate_cursor(cursor: str | None) -> None:
        if cursor is not None and (
            not isinstance(cursor, str) or not cursor.strip()
        ):
            raise MCPValidationError(
                "MCP cursor must be a non-empty string."
            )

    def _fail_protocol(self, message: str) -> None:
        self._abort()
        raise MCPProtocolError(message)

    def _abort(self) -> None:
        try:
            self._transport.close()
        except Exception:
            pass
        self._transport_opened = False
        self._state = MCPConnectionState.FAILED
