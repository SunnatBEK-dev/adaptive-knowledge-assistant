# AI SDK

A learning-focused, provider-neutral Python SDK for building production-style
AI applications from first principles. The project favors explicit contracts
and small abstractions over framework-specific magic.

## Current status

The application architecture foundation is complete through phase 3.21:

- conversation and message domain models;
- JSON repository abstraction;
- provider-neutral prompt construction;
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
regressions. Directory-backed indexes also remain synchronized across
application restarts without embedding unchanged files again.

## Architecture

```text
app/main.py
    |
RAGConversationManager
    |-- Conversation / Message
    |-- PromptBuilder -> LLMMessage
    |-- BaseLLMClient -> ClaudeClient
    |-- BaseEmbeddingClient -> SentenceTransformerEmbeddingClient
    |-- HybridRetriever -> semantic search + BM25 + rank fusion
    |                       `-- BaseVectorStore
    |                           |-- InMemoryVectorStore
    |                           `-- JsonVectorStore
    |-- Retrieval results -> Citation / RAGResponse
    |-- DirectorySynchronizer -> DocumentIngestor
    |                            `-- BaseDocumentLoader -> TextDocumentLoader
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

Indexing the same file again replaces its old chunks. Indexing a directory
performs an incremental synchronization: new and changed files are indexed,
unchanged files are skipped, and documents whose source files disappeared are
removed. Normal prompts combine embedding similarity and exact term matches,
then use the highest-ranked chunks. `/history`, `/save`, `/clear`, `/help`,
and `/exit` remain available. The first indexing run may download the
configured SentenceTransformer model.

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

The deterministic comparison dataset shows hybrid retrieval improving Hit
Rate@1, Recall@1, and MRR without a regression. Reranking is therefore deferred
until a real project dataset demonstrates a gap. The next chapter is Memory
Systems, starting with a token-aware sliding context window.
