"""
Azure / OpenAI configuration.
Supports two provider modes controlled by environment variables:

  LLM_PROVIDER=openai   → standard OpenAI API  (default, no Azure needed)
  LLM_PROVIDER=azure    → Azure OpenAI

  SEARCH_PROVIDER=local → cosine similarity on a local JSON file (default)
  SEARCH_PROVIDER=azure → Azure AI Search
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


# ── Provider mode ─────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()       # "openai" | "azure" | "qwen"
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "local").lower()  # "local"  | "azure"

# ── Standard OpenAI ───────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_GPT4O = os.getenv("OPENAI_MODEL_GPT4O", "gpt-4o")
OPENAI_MODEL_EMBEDDING = os.getenv("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large")

# ── Qwen / DashScope ──────────────────────────────────────────────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL_CHAT = os.getenv("QWEN_MODEL_CHAT", "qwen-plus")
QWEN_MODEL_EMBEDDING = os.getenv("QWEN_MODEL_EMBEDDING", "text-embedding-v2")

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT_GPT4O = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O", "gpt-4o")
AZURE_OPENAI_DEPLOYMENT_EMBEDDING = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT_EMBEDDING", "text-embedding-3-large"
)

# ── Azure AI Search ───────────────────────────────────────────────────────────
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "immunisation-guidelines")

# ── Local search ──────────────────────────────────────────────────────────────
LOCAL_CHUNKS_FILE = os.getenv("LOCAL_CHUNKS_FILE", "data/chunks_with_embeddings.json")

# ── Retrieval settings ────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 8
RETRIEVAL_SIMILARITY_THRESHOLD = 0.75


# ── Helpers used by agent modules ─────────────────────────────────────────────

def get_openai_client():
    """Return an OpenAI-compatible client based on LLM_PROVIDER."""
    from openai import OpenAI, AzureOpenAI
    if LLM_PROVIDER == "azure":
        return AzureOpenAI(
            azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
            api_key=_require("AZURE_OPENAI_API_KEY"),
            api_version=AZURE_OPENAI_API_VERSION,
        )
    elif LLM_PROVIDER == "qwen":
        return OpenAI(
            api_key=_require("DASHSCOPE_API_KEY"),
            base_url=QWEN_BASE_URL,
        )
    else:
        return OpenAI(api_key=_require("OPENAI_API_KEY"))


def get_chat_model() -> str:
    """Return the model name for chat completions."""
    if LLM_PROVIDER == "azure":
        return AZURE_OPENAI_DEPLOYMENT_GPT4O
    elif LLM_PROVIDER == "qwen":
        return QWEN_MODEL_CHAT
    return OPENAI_MODEL_GPT4O


def get_embedding_model() -> str:
    """Return the model name for embeddings."""
    if LLM_PROVIDER == "azure":
        return AZURE_OPENAI_DEPLOYMENT_EMBEDDING
    elif LLM_PROVIDER == "qwen":
        return QWEN_MODEL_EMBEDDING
    return OPENAI_MODEL_EMBEDDING
