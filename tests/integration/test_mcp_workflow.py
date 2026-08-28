import pytest

from ai_sdk.mcp import (
    MCP_PROTOCOL_VERSION,
    BaseMCPTransport,
    MCPClient,
    MCPConnectionState,
    MCPDiscoveryResult,
    MCPImplementation,
    MCPResource,
    MCPResourcePage,
    MCPServerCapabilities,
    MCPTool,
    MCPToolPage,
)


class InMemoryMCPTransport(BaseMCPTransport):
    def __init__(self):
        self.is_open = False
        self.received_meta = []

    def open(self, *, timeout_seconds):
        assert timeout_seconds == 5.0
        self.is_open = True

    def discover(self, context, *, timeout_seconds):
        assert self.is_open
        self.received_meta.append(context.to_meta())
        return MCPDiscoveryResult(
            [MCP_PROTOCOL_VERSION],
            MCPServerCapabilities(tools={}, resources={}),
            server_info=MCPImplementation("knowledge", "1.0"),
            instructions="Use the search tool before reading files.",
            ttl_ms=60_000,
            cache_scope="private",
        )

    def list_tools(self, context, *, cursor, timeout_seconds):
        assert self.is_open
        self.received_meta.append(context.to_meta())
        if cursor is None:
            return MCPToolPage(
                [
                    MCPTool(
                        "search.docs",
                        {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            },
                            "required": ["query"],
                        },
                    )
                ],
                next_cursor="page-2",
            )
        return MCPToolPage(
            [MCPTool("health-check", {"type": "object"})]
        )

    def list_resources(
        self,
        context,
        *,
        cursor,
        timeout_seconds,
    ):
        assert self.is_open
        assert cursor is None
        self.received_meta.append(context.to_meta())
        return MCPResourcePage(
            [
                MCPResource(
                    "file:///knowledge/python.md",
                    "python.md",
                    mime_type="text/markdown",
                )
            ],
            ttl_ms=10_000,
            cache_scope="private",
        )

    def close(self):
        self.is_open = False


@pytest.mark.integration
def test_stateless_mcp_discovery_and_catalog_workflow():
    transport = InMemoryMCPTransport()
    client = MCPClient(
        transport,
        client_info=MCPImplementation("ai-sdk", "0.1.0"),
        client_capabilities={"elicitation": {}},
        timeout_seconds=5,
    )

    with client:
        discovery = client.discover()
        first_tools = client.list_tools()
        second_tools = client.list_tools(
            cursor=first_tools.next_cursor
        )
        resources = client.list_resources()

        assert discovery.capabilities.supports_tools
        assert discovery.capabilities.supports_resources
        assert first_tools.tools[0].name == "search.docs"
        assert second_tools.tools[0].name == "health-check"
        assert resources.resources[0].uri == (
            "file:///knowledge/python.md"
        )

    assert client.state is MCPConnectionState.CLOSED
    assert not transport.is_open
    assert len(transport.received_meta) == 4
    assert all(
        meta["io.modelcontextprotocol/protocolVersion"]
        == MCP_PROTOCOL_VERSION
        for meta in transport.received_meta
    )
    assert all(
        meta["io.modelcontextprotocol/clientInfo"]
        == {"name": "ai-sdk", "version": "0.1.0"}
        for meta in transport.received_meta
    )
