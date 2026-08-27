from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"

CHAT_FILE = DATA_DIR / "chat.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL")

MAX_TOKENS = int(
    os.getenv("MAX_TOKENS", "1024")
)

TIMEOUT = float(
    os.getenv("TIMEOUT", "60")
)
