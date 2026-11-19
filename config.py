"""Central settings. Import this module FIRST so the offline flags are set
before transformers / huggingface_hub are imported anywhere."""
import os

# --- Run fully offline (judges test with Wi-Fi off) ---
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))

# --- Models (all open-weight, all local) ---
VL_MODEL = os.environ.get("VL_MODEL", "qwen3-vl:4b-instruct")              # image, video, documents, audio Q&A, chat
CODE_MODEL = os.environ.get("CODE_MODEL", "qwen2.5-coder:3b")     # code (use :3b if memory pressure is high)
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small")            # speech-to-text
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"             # sound classifier (AudioSet, 527 labels)

DEBUG = os.environ.get("DEBUG", "1") == "1"   # print timing for every model call

# --- Context / speed ---
NUM_CTX = 8192            # Ollama context window; default is too small for 3-page documents
IMAGE_MAX_SIDE = 1024     # resize images before sending (fewer tokens = faster)
MAX_TOKENS = 600          # hard cap on reply length: stops runaway/looping generations

# --- Thinking mode (slower, sometimes more accurate). Compare in eval. ---
THINK_IMAGE = os.environ.get("THINK_IMAGE", "0") == "1"
THINK_VIDEO = os.environ.get("THINK_VIDEO", "0") == "1"

# --- Text (documents) ---
TEXT_VERIFY = os.environ.get("TEXT_VERIFY", "1") == "1"   # second check that the quote really answers the question
QUOTE_MATCH_MIN = 85                                       # fuzzy match score for "quote is really in the document"

# --- Code ---
CODE_MAX_REPAIRS = 2      # retries when the code fails the examples in the prompt
CODE_TIMEOUT_S = 10

# --- Video ---
VIDEO_FPS = 8             # sample rate before change detection / motion tracking
VIDEO_FRAME_WIDTH = 448   # width of each frame inside the grid image
VIDEO_MAX_FRAMES = 9      # frames are tiled into ONE grid image (max 4x3) -> ~1 image worth of tokens
VIDEO_CHANGE_FRAC = 0.002 # fraction of pixels that must change to keep a frame
