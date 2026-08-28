# AI SDK

A learning-focused, provider-neutral Python SDK for building production-style
AI applications from first principles. The project favors explicit contracts
and small abstractions over framework-specific magic.

## Current status

The application architecture foundation is complete through observability
phase 9.1:

- conversation and message domain models;
- JSON repository abstraction;
- provider-neutral prompt construction;
- provider-neutral token counting with a deterministic local estimator;
- turn-aware sliding context windows with a configurable token budget;
- bounded extractive summary memory for turns outside the active window;
- explicit persistent long-term memory with lexical relevance retrieval;
- provider-neutral tool schemas with JSON Schema export;
- strict tool argument validation and allow-listed execution;
- structured tool calls and serialized success/error results;
- Claude tool-use request/response translation;
- bounded multi-round tool execution with duplicate-call protection;
- optional tool-enabled conversation and RAG orchestration;
- provider-neutral `AgentRunner` loop policy;
- explicit agent state, iteration events, and termination reasons;
- provider-neutral LLM planner with strict bounded JSON output;
- ordered plan steps with controlled status transitions and outcomes;
- bounded reflection for finished agent states and terminal plans;
- structured reflection verdicts, strengths, and improvements;
- named agent workers with explicit task assignment;
- deterministic multi-agent result collection and failure isolation;
- stateless MCP `2026-07-28` request metadata;
- provider-neutral MCP client and transport contracts;
- explicit optional server discovery and capability checks;
- ordered, cursor-based MCP tool and resource catalogs;
- MCP cache-hint preservation for list and discovery results;
- explicit MCP transport lifecycle, timeouts, and failure isolation;
- explicit MCP tool-call and resource-read contracts;
- generic MCP content plus structured tool-result preservation;
- explicitly approved MCP tool registration into the local allow-list;
- atomic rejection of MCP schemas the local validator cannot enforce;
- dependency-free stateless MCP Streamable HTTP transport;
- strict JSON-RPC response validation for JSON and request-scoped SSE;
- fresh per-request MCP authorization injection;
- safe MCP routing headers and approved `x-mcp-header` argument mirroring;
- bounded HTTP responses, disabled redirects, and contained network errors;
- manual MCP multi round-trip continuations for tools and resources;
- opaque request-state echoing with fresh JSON-RPC request identifiers;
- exact input-response matching, one-use continuations, and bounded rounds;
- provider-neutral trace and span records with validated identifiers;
- nested parent-child trace context for workflows and operations;
- bounded, thread-safe in-memory trace collection;
- opt-in tracing across LLM, retrieval, tool, agent, and MCP operations;
- safe trace metadata with sensitive-field redaction and error types only;
- Anthropic/Claude adapter behind an LLM contract;
- provider-neutral embedding-client contract;
- lazy SentenceTransformer embedding adapter;
- Document and Chunk retrieval domain models;
- deterministic character-based TextChunker with overlap;
- dependency-free cosine similarity and deterministic top-k search;
- dependency-free BM25 lexical search;
- weighted reciprocal-rank fusion for hybrid retrieval;
- provider-neutral VectorStore with in-memory and persistent JSON adapters;
- atomic document re-indexing and document-level chunk deletion;
- SemanticRetriever orchestration for indexing and search;
- retrieval-aware PromptBuilder without domain-state mutation;
- RAGConversationManager for indexing, retrieval, generation, and persistence;
- offline retrieval evaluation with Hit Rate@k, Recall@k, and MRR;
- baseline-versus-candidate retrieval comparison with metric deltas;
- RAG-enabled CLI for indexing, listing, and removing text documents;
- structured RAG responses with deterministic local source citations;
- document catalog summaries with source paths and chunk counts;
- loader-based ingestion for text files and recursive directories;
- content-hash-based incremental directory synchronization;
- stale indexed-document cleanup when source files disappear;
- conversation orchestration with rollback behavior;
- embedding cache persistence;
- isolated unit tests and opt-in integration tests.

The project now has a complete offline-tested RAG pipeline exposed through the
CLI, deterministic retrieval-quality evaluation, restart-safe local vector
persistence, document-level index lifecycle, source-aware answers, and a
loader-based ingestion layer. Normal RAG queries combine semantic similarity
with exact lexical evidence. The evaluation layer can compare this hybrid
candidate against a semantic baseline and explicitly report improvements or
regressions. Long conversations retain their full persisted history while only
the newest complete turns within the configured token budget are sent directly
to the model. Recent excluded turns are compressed into a bounded local memory
block without another API call. Directory-backed indexes also remain
synchronized across application restarts without embedding unchanged files
again. User-approved durable facts are stored separately and only relevant
matches are added to a new prompt. The SDK also has an offline-tested tool loop
that sends registered schemas to Claude, validates every requested call,
dispatches only explicitly registered Python handlers, and returns structured
results until Claude produces a final answer. The loop policy now lives in a
provider-neutral agent runtime while Claude only translates one model turn at
a time. The MCP foundation follows the stateless `2026-07-28` protocol: every
protocol request carries its own version, client identity, and capabilities.
Opening a transport does not perform a hidden handshake, and
`server/discover` remains an explicit optional call. A concrete Streamable HTTP
transport now maps each operation to a separate POST, accepts JSON or
request-scoped SSE responses, validates JSON-RPC identifiers and result
framing, and obtains authorization immediately before each request.
When a tool call or resource read needs host input, the SDK returns a local
continuation instead of answering automatically. The host can inspect the
requested interaction, gather an approved response, resume it once, or cancel
it locally. Phase 9.1 adds optional tracing to the main workflow boundaries.
Components that share one `Tracer` produce a single parent-child operation
tree with bounded timing, status, counts, and exception type metadata. Raw
prompts, model responses, tool arguments, credentials, and MCP request state
are not collected by the built-in instrumentation.

## Architecture

```text
app/main.py
    |
RAGConversationManager
    |-- Conversation / Message
    |-- PromptBuilder -> SlidingContextWindow -> LLMMessage
    |                    |-- TokenCounter -> RegexTokenCounter
    |                    `-- ExtractiveConversationSummarizer
    |-- MultiAgentCoordinator -> AgentWorker -> isolated AgentRunner run
    |-- AgentRunner -> AgentState -> AgentEvent
    |                 |-- ToolRegistry -> ToolExecutor -> ToolResult
    |                 `-- BaseToolLLMClient -> ClaudeClient (one turn)
    |-- LLMAgentPlanner -> AgentPlan -> PlanStep
    |-- LLMAgentReflector -> AgentReflection
    |-- BaseLLMClient -> ClaudeClient (plain text / streaming)
    |-- BaseEmbeddingClient -> SentenceTransformerEmbeddingClient
    |-- HybridRetriever -> semantic search + BM25 + rank fusion
    |                       `-- BaseVectorStore
    |                           |-- InMemoryVectorStore
    |                           `-- JsonVectorStore
    |-- Retrieval results -> Citation / RAGResponse
    |-- DirectorySynchronizer -> DocumentIngestor
    |                            `-- BaseDocumentLoader -> TextDocumentLoader
    |-- BaseMemoryStore -> JsonMemoryStore -> BM25 recall
    `-- ConversationRepository -> JsonConversationRepository

MCPClient -> MCPRequestContext
    `-- BaseMCPTransport -> StreamableHTTPTransport -> HTTP MCP endpoint
        |-- optional MCPDiscoveryResult
        |-- ordered MCPToolPage
        |-- ordered MCPResourcePage
        |-- MCPToolRequest -> MCPToolResult
        |-- MCPResourceReadRequest -> MCPResourceReadResult
        `-- input_required -> MCPContinuation -> continue_request

MCPToolAdapter -> approved compatible MCPTool -> ToolRegistry
    `-- ToolExecutor -> MCPClient.call_tool

Tracer -> TraceCollector -> InMemoryTraceCollector
    `-- TraceRecord (trace/span IDs, parent, timing, status, safe attributes)
```

The domain layer does not know about JSON, filesystem paths, Anthropic, API
keys, or environment variables. Provider-specific behavior stays inside the
provider adapter.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

Install local SentenceTransformer support before using `/index`:

```bash
.venv/bin/python -m pip install -e '.[embeddings]'
```

`pyproject.toml` is the dependency source of truth. `requirements.txt` and
`requirements-dev.txt` are convenience entry points.

## Configuration

Copy `.env.example` to `.env` and provide:

```text
ANTHROPIC_API_KEY=...
MODEL=...
MAX_TOKENS=1024
TIMEOUT=60
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=3
CONTEXT_TOKEN_BUDGET=3000
CONTEXT_SUMMARY_TOKEN_BUDGET=400
MEMORY_RETRIEVAL_K=3
```

Never commit `.env`, API keys, or real conversation data.

## Run the CLI

```bash
.venv/bin/python app/main.py
```

RAG document commands are:

```text
/index /absolute/path/to/guide.txt
/index /absolute/path/to/knowledge-directory
/documents
/remove doc_123456789abc
```

Long-term memory commands are:

```text
/remember Preferred language is Uzbek
/memories
/forget mem_123456789abc
```

Indexing the same file again replaces its old chunks. Indexing a directory
performs an incremental synchronization: new and changed files are indexed,
unchanged files are skipped, and documents whose source files disappeared are
removed. Normal prompts combine embedding similarity and exact term matches,
then use the highest-ranked chunks. `/history`, `/save`, `/clear`, `/help`,
and `/exit` remain available. The first indexing run may download the
configured SentenceTransformer model.

`CONTEXT_TOKEN_BUDGET` limits conversation memory sent to the model. Selection
keeps complete recent user/assistant turns and always retains the newest turn,
even when that turn alone exceeds the soft budget. `/history` and JSON
persistence continue to keep the complete conversation.
`CONTEXT_SUMMARY_TOKEN_BUDGET` limits the deterministic extractive memory made
from excluded turns. This local summarizer adds no model request, token charge,
or network latency.

Long-term memory is explicit: the application never guesses which facts to
store. `/remember` persists a user-approved fact in `data/memories.json`, and
lexical retrieval adds only matching memories to later prompts. `/clear`
removes conversation history but leaves long-term memory intact; use `/forget`
to delete a stored fact.

The tool layer supports string, integer, number, and boolean parameters,
required and optional arguments, deterministic JSON Schema export, duplicate
registration protection, strict unknown-argument rejection, and contained
handler errors. It does not use dynamic imports, `eval`, or shell execution.
When a `ToolExecutor` is configured on `ConversationManager` or
`RAGConversationManager`, Claude may request one or more registered tools in a
round. Results, including contained validation and execution errors, are sent
back until Claude returns final text. `AgentRunner` defaults to at most eight
tool rounds and rejects duplicate call IDs. Its returned `AgentState` contains
ordered `AgentEvent` records and ends with either `final_response` or
`max_tool_rounds`. Tool traces are transient when using the conversation
manager; persisted history contains the original user message and final
assistant answer. Applications that need the trace can invoke
`AgentRunner.run()` directly and optionally receive each event through a
callback. Tool-enabled streaming is intentionally unsupported for now, so use
`send_message()` for this workflow. The SDK does not register application
tools automatically—the host application owns that allow-list.

`LLMAgentPlanner` is an opt-in provider-neutral planning layer. It asks any
`BaseLLMClient` for a bounded JSON list of unique steps and converts that list
into an `AgentPlan`. Plans expose only controlled sequential transitions from
`pending` to `in_progress`, then `completed` or `failed`, with an optional
outcome stored for each completed step. Planning does not automatically execute
steps or re-plan: the host application explicitly passes each active step to
`AgentRunner`, which keeps planning policy separate from tool execution.

`LLMAgentReflector` performs one opt-in review of a finished `AgentState` or a
completed/failed `AgentPlan`. It returns a strict verdict (`passed`,
`needs_improvement`, or `failed`) plus bounded lists of strengths and
improvements. Reflection never mutates the reviewed object, executes tools,
retries work, or applies its own suggestions. Snapshot input is capped and may
be truncated; because it is sent to the configured LLM provider, host
applications should not include secrets in state, plan outcomes, or tool
results.

`MultiAgentCoordinator` is an explicit sequential coordinator. Named workers
are registered with one responsibility and one `AgentRunner`; every task names
its worker directly. The coordinator validates all assignments before any work
starts, creates a fresh `AgentState` for each task, preserves input order in
the result, and contains one worker failure so later tasks can continue.
Exception messages are not exposed. Isolation is at the agent-state level, not
an operating-system process boundary, so stateful clients and tool handlers
remain the host application's responsibility. The coordinator does not perform
automatic delegation, implicit handoffs, shared-state mutation, parallel
execution, or recursive agent calls.

`MCPClient` provides the local contract for MCP `2026-07-28`. It has an
explicit `new -> opening -> open -> failed/closed` transport lifecycle, sends
protocol version, client identity, and client capabilities with every protocol
request, and contains transport exception messages. `server/discover` is
optional and never runs automatically. If the host calls it, the client checks
the requested protocol version and advertised tool/resource capabilities.
`tools/list` and `resources/list` return exactly one ordered page at a time,
including the next cursor and server cache hints; the host decides whether to
request another page or cache a result. `tools/call` preserves text, non-text,
structured, and remote error results. `resources/read` preserves multiple text
or binary content items plus cache hints.

`tools/call` and `resources/read` may instead return `MCPContinuation` when the
server responds with `resultType: input_required`. The continuation exposes
the requested elicitation, sampling, or roots operations but keeps the
server's `requestState` opaque. `continue_request()` requires responses whose
keys exactly match the outstanding requests, echoes the state unchanged, and
uses a new JSON-RPC request ID. Each continuation is consumed once;
`cancel_continuation()` discards it locally. A logical request defaults to at
most ten input-required rounds, and every network retry uses the client's
normal request timeout. No server request is answered automatically.

`StreamableHTTPTransport` is the dependency-free concrete network adapter. It
sends every operation as a new POST with matching protocol metadata and
`MCP-Protocol-Version`, `Mcp-Method`, and, where required, `Mcp-Name` headers.
It supports ordinary JSON responses and request-scoped SSE, enforces matching
JSON-RPC IDs, bounds response size, rejects redirects, and converts valid
remote JSON-RPC errors into typed `MCPRemoteError` values. An optional
authorization callback is invoked separately for every request so the host can
provide a current token without storing it in the transport.

Tools may mark compatible primitive input properties with `x-mcp-header`.
Those values are mirrored as `Mcp-Param-*` headers when the tool is called;
non-ASCII values use the protocol's base64 sentinel form. Tools with invalid
or unreachable header annotations are excluded before approval and
registration.

```python
transport = StreamableHTTPTransport(
    "https://example.com/mcp",
    authorization_provider=lambda: get_current_authorization(),
)
client = MCPClient(
    transport,
    client_info=MCPImplementation("my-client", "1.0"),
)

with client:
    tools = client.list_tools()
    outcome = client.call_tool("confirmable_action", {"value": 1})
    if isinstance(outcome, MCPContinuation):
        outcome = client.continue_request(
            outcome,
            {"confirm": {"action": "accept", "content": {}}},
        )
```

`MCPToolAdapter` never discovers or registers tools automatically. The host
must pass exact approved names. Only schemas that fit the local tool layer's
string, integer, number, and boolean parameter subset are accepted; unsupported
constraints, nested inputs, incompatible names, missing approvals, and
registry collisions are rejected before any registry change. Approved remote
errors remain explicit tool errors, while transport exceptions still expose
only their exception type. The automatic local tool adapter does not collect
interactive input: it cancels an unexpected continuation and returns a safe
tool error so the host can use the manual client API instead.

This phase deliberately has no automatic OAuth discovery or token refresh,
automatic pagination/discovery, legacy MCP protocol fallback, subscriptions,
or automatic multi round-trip fulfillment. The authorization callback lets
the host own credential refresh without exposing secrets to the SDK.

## Tracing

Tracing is opt-in. Create one collector and tracer, then pass the same tracer
to the top-level manager or lower-level component that should be observed:

```python
from ai_sdk.observability import InMemoryTraceCollector, Tracer

collector = InMemoryTraceCollector(max_records=1000)
tracer = Tracer(collector)

manager = RAGConversationManager(
    conversation=conversation,
    prompt_builder=prompt_builder,
    client=llm_client,
    repository=repository,
    chunker=chunker,
    retriever=retriever,
    tracer=tracer,
)
response = manager.send_message("question")

for record in collector.records():
    print(record.to_dict())
```

The built-in instrumentation records operation names, parent-child
relationships, elapsed time, outcome status, safe counts, and exception class
names. It does not record exception messages or application content. Attribute
names associated with prompts, content, credentials, tokens, tool arguments,
and MCP state are redacted; custom instrumentation must still use deliberate,
non-sensitive attribute names. Records stay in the bounded in-memory collector
unless the host supplies another `TraceCollector` implementation. This phase
does not yet include remote trace propagation, sampling, an OpenTelemetry
exporter, metrics, logs, or persistent trace reports.

The default ingestion layer supports UTF-8 `.txt`, `.md`, `.markdown`, and
`.rst` files. Directory synchronization scans recursively in deterministic
path order, skips unsupported formats, and persists content hashes plus the
owning root directory. `/documents` shows each source path and its current
chunk count.

After each RAG answer, the CLI prints numbered sources with document ID, chunk
ID, and fused retrieval score. Local source paths are mapped to citations after
the model call and are not included in the Claude prompt.

## Tests

The default command runs deterministic unit tests only and never calls an
external API:

```bash
.venv/bin/python -m pytest
```

Measure unit-test coverage when the test extras are installed:

```bash
.venv/bin/python -m pytest \
  --cov=ai_sdk \
  --cov-report=term-missing
```

Run offline integration tests explicitly:

```bash
.venv/bin/python -m pytest -m integration
```

The real Anthropic smoke test is both marked `external` and guarded by an
environment flag. It may consume API tokens and must be explicitly enabled:

```bash
RUN_ANTHROPIC_INTEGRATION=1 \
  .venv/bin/python -m pytest \
  -m 'integration and external' \
  tests/integration/test_claude_api.py
```

## Next chapter

Phase 9.2 will add a small provider-neutral evaluation harness: explicit eval
cases, deterministic evaluator contracts, pass/fail criteria, and aggregate
quality reports. Cost and latency summaries, regression gates, exporters, and
persistent observability storage remain later opt-in layers rather than core
runtime requirements.
