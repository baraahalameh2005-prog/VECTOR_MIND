"""
embedder.py — EMBED stage: turns text into vectors (embeddings).

Model: intfloat/multilingual-e5-small (Arabic + English + 100 other languages),
run locally on the CPU with ONNX Runtime - no internet needed after download.

How one text becomes one vector (official steps from the model card):
    1. Prefix:     "passage: " for document chunks, "query: " for questions
    2. Tokenize:   split into tokens (max 512), pad shorter texts in a batch
    3. Model:      one vector (384 numbers) for EACH token
    4. Mean pool:  average the token vectors, ignoring padding (attention mask)
    5. Normalize:  scale the vector to length 1

E5Embedder is a Chroma "embedding function": Chroma calls it automatically
    - documents (collection.add)   -> __call__     -> "passage: " prefix
    - questions (collection.query) -> embed_query  -> "query: " prefix

get_embedder() loads the model ONCE and returns the same object every time.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from chromadb.api.types import EmbeddingFunction
from tokenizers import Tokenizer

from config import EMBEDDING_MODEL_DIR, EMBEDDING_MAX_TOKENS


class E5Embedder(EmbeddingFunction):

    def __init__(self, model_dir=EMBEDDING_MODEL_DIR):
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "onnx" / "model.onnx"
        tokenizer_path = self.model_dir / "onnx" / "tokenizer.json"

        if not model_path.is_file() or not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Embedding model not found in {self.model_dir}. "
                "Run once (needs internet):  python download_model.py"
            )

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=EMBEDDING_MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")

    # ---------- called by Chroma ----------
    def __call__(self, input):
        """Document chunks -> vectors."""
        return self._embed(["passage: " + text for text in input])

    def embed_query(self, input):
        """Questions -> vectors."""
        return self._embed(["query: " + text for text in input])

    # ---------- the 5 steps ----------
    def _embed(self, texts, batch_size=16):
        vectors = []

        for start in range(0, len(texts), batch_size):
            encodings = self.tokenizer.encode_batch(texts[start:start + batch_size])
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            token_vectors = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": np.zeros_like(input_ids),
            })[0]                                               # shape: (texts, tokens, 384)

            mask = attention_mask[:, :, None]                   # 1 = real token, 0 = padding
            mean = (token_vectors * mask).sum(axis=1) / mask.sum(axis=1)
            mean = mean / np.linalg.norm(mean, axis=1, keepdims=True)
            vectors.extend(mean)

        return vectors

    # ---------- required by Chroma's EmbeddingFunction interface ----------
    @staticmethod
    def name():
        return "multilingual-e5-small-onnx"

    def get_config(self):
        return {"model_dir": str(self.model_dir)}

    @staticmethod
    def build_from_config(config):
        # Chroma calls this several times while creating a collection.
        # Return the already-loaded model instead of loading 470 MB again.
        return get_embedder()


# ============================================================
# Load the model once, reuse it everywhere
# ============================================================
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = E5Embedder()
    return _embedder
