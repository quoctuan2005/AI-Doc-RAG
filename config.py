import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "google")   # "google" | "openai"
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL       = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ── Embedding ────────────────────────────────────────────────
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ────────────────────────────────────────────────
TOP_K              = int(os.getenv("TOP_K", "5"))

# ── Supported file types ─────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}
