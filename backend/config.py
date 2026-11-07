"""
config.py — All tunable settings for the RAG backend, in one place.

Every other backend file imports its settings from here, so changing a value
(e.g. CHUNK_SIZE) never requires editing the pipeline code itself.
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent      # VECTOR_MIND/
ORIGINAL_DOCS_DIR = PROJECT_ROOT / "documents" / "original"
PROCESSED_DOCS_DIR = PROJECT_ROOT / "documents" / "processed"

# ============================================================
# Chunking  (starting values from the Bootcamp — to be tested)
# ============================================================
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # characters shared between neighbouring chunks

# ============================================================
# Embeddings  (multilingual-e5-small, run locally with ONNX - no internet needed)
# ============================================================
# Chosen after measuring: Arabic Hit@3 rose from 1/17 (Chroma's default
# all-MiniLM-L6-v2) to 14/17. Download once with:  python download_model.py
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"   # pinned version
EMBEDDING_MODEL_DIR = PROJECT_ROOT / "models" / "multilingual-e5-small"
EMBEDDING_MAX_TOKENS = 512      # model limit; longer text is cut

# ============================================================
# Retrieval
# ============================================================
COLLECTION_NAME = "documents"   # name of the Chroma collection
TOP_K = 5                       # number of chunks to retrieve per question

# Refusal threshold — intentionally NOT set yet.
# Decided in Phase 3 after printing real distances for
# answerable and unanswerable questions.
MAX_DISTANCE = None

# ============================================================
# Generation (LLM)  — local Ollama only (no paid APIs)
# ============================================================
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"           # pulled with:  ollama pull qwen2.5:7b   (fallback: "qwen2.5:3b")
LLM_TIMEOUT_SECONDS = 180
MAX_TOKENS = 250
NUM_CTX = 2048                      # context window sent to Ollama (measured prompt with 4 chunks fits; smaller = faster on CPU)

# Exact refusal sentence — generator prompt and rag_pipeline both use it.
REFUSAL_TEXT = "لا أجد هذه المعلومة في المستندات المتاحة لي."
