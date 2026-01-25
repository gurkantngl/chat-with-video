from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
LANCEDB_DIR = DATA_DIR / "lancedb"

DEFAULT_SAMPLE_FPS = 1
DEFAULT_MAX_FRAMES = None

FLORENCE_MODEL_NAME = "microsoft/Florence-2-base"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
