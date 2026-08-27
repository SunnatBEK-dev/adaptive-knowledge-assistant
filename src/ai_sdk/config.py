from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"

CHAT_FILE = DATA_DIR / "chat.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"
VECTOR_STORE_FILE = DATA_DIR / "vectors.json"

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL")

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
