# AI SDK

A learning-focused, provider-neutral Python SDK for building production-style
AI applications from first principles. The project favors explicit contracts
and small abstractions over framework-specific magic.

## Current status

The application architecture foundation is complete through phase 3.12:

- conversation and message domain models;
- JSON repository abstraction;
- provider-neutral prompt construction;
- Anthropic/Claude adapter behind an LLM contract;
- conversation orchestration with rollback behavior;
- embedding cache persistence;
- isolated unit tests and opt-in integration tests.

Retrieval and RAG are intentionally the next chapter, not part of the current
foundation.

## Architecture

```text
app/main.py
    |
ConversationManager
    |-- Conversation / Message
    |-- PromptBuilder -> LLMMessage
    |-- BaseLLMClient -> ClaudeClient
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

`pyproject.toml` is the dependency source of truth. `requirements.txt` and
`requirements-dev.txt` are convenience entry points.

## Configuration

Copy `.env.example` to `.env` and provide:

```text
ANTHROPIC_API_KEY=...
MODEL=...
MAX_TOKENS=1024
TIMEOUT=60
```

Never commit `.env`, API keys, or real conversation data.

## Run the CLI

```bash
.venv/bin/python app/main.py
```

Available commands are `/exit`, `/save`, `/clear`, and `/history`.

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

The planned Retrieval/RAG sequence is:

1. embedding-client abstraction;
2. document and chunk domain models;
3. deterministic chunking;
4. cosine similarity and top-k retrieval;
5. vector-store and retriever contracts;
6. retrieval-aware prompt construction;
7. RAG orchestration and evaluation.
