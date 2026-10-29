"""Central settings for the AIS Website Assistant.

Every tunable value lives here so the agents contain no magic numbers.
Anything that a user might reasonably change can be overridden with an
environment variable, e.g.  OLLAMA_MODEL=llama3.2:3b python -m ...
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                 # committed: the saved copy of the website
PAGES_FILE = RAW_DIR / "pages.jsonl"       # one JSON object per crawled page
MANIFEST_FILE = RAW_DIR / "manifest.json"  # how/when the snapshot was made
VECTOR_DIR = DATA_DIR / "chroma"           # rebuilt from PAGES_FILE, not committed

# --------------------------------------------------------------------------- #
# Crawler (Step 1)
# --------------------------------------------------------------------------- #
SITE_URL = "https://aisinfo.com/"
# Other AIS-owned hosts we also read. Each has its own robots.txt, which is obeyed separately.
EXTRA_HOSTS = ("careers.aisinfo.com",)
USER_AGENT = "AIS-Assistant-Crawler/0.1 (interview take-home; polite, max 1 request/second)"
MIN_CRAWL_DELAY = 1.0      # seconds between requests. The brief says "one per second or slower".
MAX_PAGES = 150            # safety cap; a company site is far smaller than this
REQUEST_TIMEOUT = 15       # seconds
MAX_RETRIES = 3
MIN_PAGE_CHARS = 200       # shorter pages are almost always empty / JS-only shells

# --------------------------------------------------------------------------- #
# Chunking and search (Step 2)
# --------------------------------------------------------------------------- #
CHUNK_MAX_CHARS = 900      # ~200 tokens: one idea per chunk, far below the embedder's 512-token limit
CHUNK_MIN_CHARS = 150      # a smaller leftover is merged into the previous chunk
OVERLAP_MAX_CHARS = 120    # the last line of a chunk is repeated in the next one if this short (a heading)
COLLECTION_NAME = "ais_site"
TOP_K = 6                  # passages handed to the answer agent (~1,400 tokens)

# --------------------------------------------------------------------------- #
# Models. All local and open source.
# --------------------------------------------------------------------------- #
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge models are trained to see this prefix on SEARCH QUERIES (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "90"))  # seconds per call; a cold model load can take ~20 s
LLM_RETRIES = 1            # one retry on timeout / connection error, then the orchestrator falls back
LLM_KEEP_ALIVE = "30m"     # keep the model in memory between visitors
LLM_NUM_CTX = 4096        # Ollama's default (2048) is too small for 5 passages + instructions

# --------------------------------------------------------------------------- #
# Agents and conversation (Steps 3-6)
# --------------------------------------------------------------------------- #
# Passages scoring below this are treated as "not found". bge-small scores are compressed:
# unrelated text still scores ~0.45, a real match is usually 0.65+. Tuned on the test set (see eval/).
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.55"))
MAX_ANSWER_SENTENCES = 3   # the brief asks for short answers; enforced in code, not left to the model
HISTORY_TURNS = 3         # earlier (question, answer) pairs the router sees when resolving a follow-up
CONTACT_URL = "https://aisinfo.com/contact-us"
CAREERS_URL = "https://careers.aisinfo.com/"

# --------------------------------------------------------------------------- #
# Evaluation (Step 7)
# --------------------------------------------------------------------------- #
EVAL_DIR = ROOT_DIR / "eval"
TEST_SET_FILE = EVAL_DIR / "test_set.json"
