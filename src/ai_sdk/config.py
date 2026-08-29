from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"

CHAT_FILE = DATA_DIR / "chat.json"
DIRECT_CHAT_DIR = DATA_DIR / "direct_chat"
SUPER_AI_CHAT_FILE = DATA_DIR / "super_ai_chat.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"
VECTOR_STORE_FILE = DATA_DIR / "vectors.json"
MEMORY_FILE = DATA_DIR / "memories.json"

AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic")
MODEL = os.getenv("MODEL")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL") or MODEL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or MODEL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or MODEL

# Backward-compatible alias for the original Claude-only configuration.
API_KEY = ANTHROPIC_API_KEY

MAX_TOKENS = int(
    os.getenv("MAX_TOKENS", "1024")
)

TIMEOUT = float(
    os.getenv("TIMEOUT", "60")
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "500")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "50")
)

RETRIEVAL_K = int(
    os.getenv("RETRIEVAL_K", "3")
)

CONTEXT_TOKEN_BUDGET = int(
    os.getenv("CONTEXT_TOKEN_BUDGET", "3000")
)

CONTEXT_SUMMARY_TOKEN_BUDGET = int(
    os.getenv("CONTEXT_SUMMARY_TOKEN_BUDGET", "400")
)

MEMORY_RETRIEVAL_K = int(
    os.getenv("MEMORY_RETRIEVAL_K", "3")
)

LLM_RETRY_MAX_ATTEMPTS = int(
    os.getenv("LLM_RETRY_MAX_ATTEMPTS", "3")
)

LLM_RETRY_INITIAL_DELAY = float(
    os.getenv("LLM_RETRY_INITIAL_DELAY", "0.25")
)

LLM_RETRY_MAX_DELAY = float(
    os.getenv("LLM_RETRY_MAX_DELAY", "2.0")
)
