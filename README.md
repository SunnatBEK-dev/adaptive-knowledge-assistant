# AI SDK

A learning-focused, provider-neutral Python SDK for building production-style
AI applications from first principles. The project favors explicit contracts
and small abstractions over framework-specific magic.

## Current status

The application architecture foundation is complete through phase 3.16:

- conversation and message domain models;
- JSON repository abstraction;
- provider-neutral prompt construction;
- Anthropic/Claude adapter behind an LLM contract;
- provider-neutral embedding-client contract;
- lazy SentenceTransformer embedding adapter;
- Document and Chunk retrieval domain models;
- deterministic character-based TextChunker with overlap;
- dependency-free cosine similarity and deterministic top-k search;
- provider-neutral VectorStore with in-memory and persistent JSON adapters;
- atomic document re-indexing and document-level chunk deletion;
- SemanticRetriever orchestration for indexing and search;
- retrieval-aware PromptBuilder without domain-state mutation;
- RAGConversationManager for indexing, retrieval, generation, and persistence;
- offline retrieval evaluation with Hit Rate@k, Recall@k, and MRR;
- RAG-enabled CLI for indexing, listing, and removing text documents;
- conversation orchestration with rollback behavior;
- embedding cache persistence;
- isolated unit tests and opt-in integration tests.

The project now has a complete offline-tested RAG pipeline exposed through the
CLI, deterministic retrieval-quality evaluation, restart-safe local vector
persistence, and a document-level index lifecycle.

## Architecture

```text
app/main.py
    |
RAGConversationManager
    |-- Conversation / Message
    |-- PromptBuilder -> LLMMessage
    |-- BaseLLMClient -> ClaudeClient
    |-- BaseEmbeddingClient -> SentenceTransformerEmbeddingClient
    |-- SemanticRetriever -> BaseVectorStore
    |                          |-- InMemoryVectorStore
    |                          `-- JsonVectorStore
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
/documents
/remove doc_123456789abc
```

Indexing the same path again replaces its old chunks. Normal prompts use the
most relevant indexed chunks. `/history`, `/save`, `/clear`, `/help`, and
`/exit` remain available. The first indexing run may download the configured
SentenceTransformer model.

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

The next Retrieval/RAG phase is source-aware answers: preserve retrieval source
details and return citations alongside generated responses.
