"""
config.py
=========
Settings shared by every script. Servers, model names and keys come from environment variables (a .env file
in this folder is read if present), so nothing here is tied to one machine. Every model is reached through an
OpenAI-compatible endpoint (vLLM, llama.cpp's llama-server, or OpenRouter).
"""
import os
from datetime import datetime

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# --- Benchmark definition -------------------------------------------------------------------------------------
CUT_OFF_DATE = datetime(2025, 3, 31, 23, 59, 59)       # C_pre: newest revision at or before; C_post: created after
MAX_CHUNK_SIZE = 1500                                   # characters per chunk
CHUNK_OVERLAP = 250
RETRIEVAL_TOP_K = 5                                     # chunks given to the model (single-shot RAG)
RRF_K = 60                                              # reciprocal rank fusion constant
GENERATION_MAX_TOKENS = int(os.environ.get("GENERATION_MAX_TOKENS", 16000))   # answer budget, reasoning included

# --- Files ----------------------------------------------------------------------------------------------------
DATA_DIR = os.environ.get("DC_DATA_DIR", "data")
TRAIN_JSON_FILE = os.environ.get("DC_PRE_JSON", os.path.join(DATA_DIR, "c_pre.json"))     # C_pre pages
TEST_JSON_FILE = os.environ.get("DC_POST_JSON", os.path.join(DATA_DIR, "c_post.json"))    # C_post pages
CHROMA_DB_PATH = os.environ.get("DC_CHROMA_DB", os.path.join(DATA_DIR, "chroma"))
TRAIN_COLLECTION_NAME = "minecraft_wiki_data_Mar2025"          # C_pre chunks (retrievable)
TEST_COLLECTION_NAME = "minecraft_wiki_data_latest_mar2025"    # C_post chunks (question generation only)
QUERY_REWRITE_CACHE = os.environ.get("DC_REWRITE_CACHE", os.path.join(DATA_DIR, "query_rewrites.json"))

# --- Endpoints ------------------------------------------------------------------------------------------------
# Embeddings: Qwen3-Embedding-8B, Q8_0 GGUF, served by llama.cpp's llama-server (OpenAI-compatible /v1/embeddings).
EMBED_URL = os.environ.get("DC_EMBED_URL", "http://localhost:8001/v1")
EMBED_MODEL = os.environ.get("DC_EMBED_MODEL", "qwen3-embedding:8b-q8_0")
# Cross-encoder reranker: nvidia/llama-nemotron-rerank-1b-v2 served by vLLM (/rerank or /score).
RERANKER_BASE_URL = os.environ.get("DC_RERANK_URL", "http://localhost:8003/v1")
RERANKER_MODEL = os.environ.get("DC_RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
# Query rewriting (pipelines hybrid_rewrite and full): Qwen3 30B-A3B with thinking disabled in the paper.
QUERY_REWRITE_BASE_URL = os.environ.get("DC_REWRITE_URL", "")
QUERY_REWRITE_MODEL = os.environ.get("DC_REWRITE_MODEL", "")
# Default endpoint for generation and judging when a script is not given --url.
LLM_URL = os.environ.get("DC_LLM_URL", "http://localhost:8000/v1")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENAI_API_KEY = os.environ.get("DC_API_KEY", "EMPTY")              # key for self-hosted endpoints, if any
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def get_node_key(url: str) -> str:
    """API key for an endpoint URL: the OpenRouter key for OpenRouter, DC_API_KEY otherwise."""
    return OPENROUTER_API_KEY if "openrouter" in (url or "").lower() else OPENAI_API_KEY


def make_client(url: str, timeout: float = 600.0) -> OpenAI:
    return OpenAI(base_url=url, api_key=get_node_key(url) or "EMPTY", timeout=timeout, max_retries=0)


os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
