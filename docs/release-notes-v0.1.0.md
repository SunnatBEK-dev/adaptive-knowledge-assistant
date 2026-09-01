# Adaptive Knowledge Assistant v0.1.0

The first portfolio release turns the underlying provider-neutral AI SDK into
a focused local knowledge-assistant product.

## Highlights

- Upload PDF, Markdown, or TXT and ask grounded questions with local source
  cards and page-aware PDF citations.
- Choose one configured provider or use an explainable Adaptive Multi-Model
  route across Anthropic, OpenAI, and Gemini.
- Inspect route and stage progress over SSE, cancel safely at workflow
  boundaries, and keep API keys out of responses and metrics.
- Reproduce a 27-case hybrid-retrieval benchmark: Hit Rate@3 1.000, Recall@3
  1.000, and MRR 0.963.
- Reproduce a 24/24 routing benchmark with 48 estimated provider requests
  versus a 72-request always-FULL baseline.
- Run 798 offline tests under a 95% coverage gate; the measured v0.1.0 branch
  coverage is 96.69% on Python 3.14.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[embeddings,documents,web]'
cp .env.example .env
adaptive-knowledge
```

This release is a local, single-user BYOK demonstration. It intentionally does
not claim hosted-production readiness, general-domain answer quality, or
multi-tenant security.
