from abc import ABC, abstractmethod

from ai_sdk.mcp.model import (
    MCPDiscoveryResult,
    MCPRequestContext,
    MCPResourcePage,
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
    def close(self) -> None:
        """Release local transport resources."""
