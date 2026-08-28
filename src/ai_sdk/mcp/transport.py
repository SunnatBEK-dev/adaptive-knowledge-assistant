from abc import ABC, abstractmethod

from ai_sdk.mcp.model import (
    MCPDiscoveryResult,
    MCPRequestContext,
    MCPResourceReadRequest,
    MCPResourceReadResult,
    MCPResourcePage,
    MCPToolRequest,
    MCPToolResult,
    MCPToolPage,
)


class BaseMCPTransport(ABC):
    """Provider-neutral synchronous MCP transport boundary."""

    @abstractmethod
    def open(self, *, timeout_seconds: float) -> None:
        """Prepare the transport for requests."""

    @abstractmethod
    def discover(
        self,
        context: MCPRequestContext,
        *,
        timeout_seconds: float,
    ) -> MCPDiscoveryResult:
        """Perform the optional server/discover request."""

    @abstractmethod
    def list_tools(
        self,
        context: MCPRequestContext,
        *,
        cursor: str | None,
        timeout_seconds: float,
    ) -> MCPToolPage:
        """Return one ordered tools/list page."""

    @abstractmethod
    def list_resources(
        self,
        context: MCPRequestContext,
        *,
        cursor: str | None,
        timeout_seconds: float,
    ) -> MCPResourcePage:
        """Return one ordered resources/list page."""

    @abstractmethod
    def call_tool(
        self,
        context: MCPRequestContext,
        request: MCPToolRequest,
        *,
        timeout_seconds: float,
    ) -> MCPToolResult:
        """Perform one complete tools/call request."""

    @abstractmethod
    def read_resource(
        self,
        context: MCPRequestContext,
        request: MCPResourceReadRequest,
        *,
        timeout_seconds: float,
    ) -> MCPResourceReadResult:
        """Perform one complete resources/read request."""

    @abstractmethod
    def close(self) -> None:
        """Release local transport resources."""
