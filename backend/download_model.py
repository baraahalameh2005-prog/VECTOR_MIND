"""
download_model.py — download the embedding model ONCE (needs internet), then
the project runs fully offline.

Usage (from backend/, with the venv active):
    python download_model.py

- Downloads a pinned version (EMBEDDING_MODEL_REVISION) of multilingual-e5-small.
- Checks the SHA-256 fingerprint of every file, so every team member gets
  exactly the same model. A wrong fingerprint stops with an error.
- Files go to models/multilingual-e5-small/ (ignored by Git - about 490 MB).
- Safe to run again: it re-checks the files.
"""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

from config import EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION

# file path inside the model repo -> expected SHA-256
FILES = {
    "onnx/model.onnx":              "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    "onnx/tokenizer.json":          "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    "onnx/config.json":             "bbb7c1333fc4b3e27fbc9cd5d2070aabcc1d4dfb99917c3633e772f97545a6b6",
    "onnx/tokenizer_config.json":   "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
    "onnx/special_tokens_map.json": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    "1_Pooling/config.json":        "987f7a67a38fa564c849bb5d277c52ab9088a84368fc0be31a354125aebb12a0",
    "modules.json":                 "c6e29747481e8b5dd2b58401966aeac910de39092f90cda9a704b1545f902b04",
    "sentence_bert_config.json":    "948201d8329907aae938fa62f9ceeed53f5694dacc2b87b9f3b78b37ee986529",
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model():
    print(f"Model: {EMBEDDING_MODEL_NAME} @ {EMBEDDING_MODEL_REVISION[:8]}")
    print(f"Into : {EMBEDDING_MODEL_DIR}\n")

    for filename, expected in FILES.items():
        path = Path(hf_hub_download(
            repo_id=EMBEDDING_MODEL_NAME,
            filename=filename,
            revision=EMBEDDING_MODEL_REVISION,
            local_dir=EMBEDDING_MODEL_DIR,
        ))

        actual = sha256_of(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}. "
                             "Delete the file and run again.")

        print(f"[OK] {filename:30s} {path.stat().st_size / 1e6:8.1f} MB  sha256 verified")

    print("\nDone - the model now works offline.")


if __name__ == "__main__":
    download_model()
