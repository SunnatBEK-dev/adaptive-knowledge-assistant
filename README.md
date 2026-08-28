# AI SDK

A learning-focused, provider-neutral Python SDK for building production-style
AI applications from first principles. The project favors explicit contracts
and small abstractions over framework-specific magic.

## Current status

The application architecture foundation is complete through Agents and MCP
phase 8.1:

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
`server/discover` remains an explicit optional call.

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
    `-- BaseMCPTransport
        |-- optional MCPDiscoveryResult
        |-- ordered MCPToolPage
        `-- ordered MCPResourcePage
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
request another page or cache a result.

This phase deliberately has no concrete network or subprocess transport,
credential handling, remote tool execution, resource reading, automatic
pagination, or automatic discovery. The transport abstraction makes those
additions testable without coupling the domain layer to one MCP library.

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

MCP phase 8.2 will add explicit remote tool-call and resource-read contracts,
then adapt approved MCP tools into the SDK's existing allow-listed tool layer.
A concrete transport and credentials remain separate follow-up work so network
and security policy do not leak into the protocol domain models.
