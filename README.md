# AI SDK

A learning-focused, provider-neutral Python SDK for building production-style
AI applications from first principles. The project favors explicit contracts
and small abstractions over framework-specific magic.

## Current status

The application architecture foundation is complete through Tool Calling phase 5.1:

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
matches are added to a new prompt. The SDK also has an offline-tested tool
foundation that validates every call before dispatching only explicitly
registered Python handlers.

## Architecture

```text
app/main.py
    |
RAGConversationManager
    |-- Conversation / Message
    |-- PromptBuilder -> SlidingContextWindow -> LLMMessage
    |                    |-- TokenCounter -> RegexTokenCounter
    |                    `-- ExtractiveConversationSummarizer
    |-- BaseLLMClient -> ClaudeClient
    |-- BaseEmbeddingClient -> SentenceTransformerEmbeddingClient
    |-- HybridRetriever -> semantic search + BM25 + rank fusion
    |                       `-- BaseVectorStore
    |                           |-- InMemoryVectorStore
    |                           `-- JsonVectorStore
    |-- Retrieval results -> Citation / RAGResponse
    |-- DirectorySynchronizer -> DocumentIngestor
    |                            `-- BaseDocumentLoader -> TextDocumentLoader
    |-- BaseMemoryStore -> JsonMemoryStore -> BM25 recall
    |-- ToolRegistry -> ToolExecutor -> ToolResult
    `-- ConversationRepository -> JsonConversationRepository
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
Claude tool-use blocks are not connected to the application loop yet.

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

The next Tool Calling phase is provider integration: translate schemas to the
Claude request format, parse tool-use blocks into provider-neutral `ToolCall`
objects, execute them, and return `ToolResult` blocks until the model produces
a final text response.
