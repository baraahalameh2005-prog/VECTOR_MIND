"""
vector_store.py — EMBED & STORE + RETRIEVE stages of the RAG pipeline.

Everything that touches Chroma lives in this file.

Input  (from text_chunker.chunk_pages):
    [{"text": "...", "source": "file.pdf", "page": 1, "chunk_index": 1}, ...]

build_collection(chunks)  -> stores the chunks in an in-memory Chroma collection.
                             Chroma computes the embeddings automatically (as in
                             the Bootcamp), using OUR embedding function
                             (multilingual-e5-small, see embedder.py) instead of
                             Chroma's default English model.
search(collection, question)
                          -> top-k closest chunks, each with its metadata and distance:
    [{"text": "...", "source": "file.pdf", "page": 1, "chunk_index": 1, "distance": 0.83}, ...]

Notes:
- In-memory: the collection is rebuilt every time the program starts.
- Distance: Chroma's default metric (L2). Smaller distance = closer meaning.
- The embedding model is loaded once (get_embedder) and works offline.
"""

import chromadb

from config import COLLECTION_NAME, TOP_K, MAX_DISTANCE
from embedder import get_embedder


# ============================================================
# Unique id for each chunk, e.g. "guide.pdf-p3-c2"
# ============================================================
def make_chunk_id(chunk):
    return f"{chunk['source']}-p{chunk['page']}-c{chunk['chunk_index']}"


# ============================================================
# EMBED & STORE: chunks -> Chroma collection
# ============================================================
def build_collection(chunks, name=COLLECTION_NAME):
    """Create a fresh collection and add all chunks to it.

    If a collection with the same name already exists, it is deleted first,
    so old chunks never mix with new ones.
    """
    if not chunks:
        raise ValueError("No chunks to store.")

    ids = [make_chunk_id(chunk) for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk ids found - two files may have the same name.")

    client = chromadb.Client()

    existing_names = [collection.name for collection in client.list_collections()]
    if name in existing_names:
        client.delete_collection(name)
        print(f"[INFO] collection '{name}' already existed - deleted and rebuilt")

    collection = client.create_collection(name, embedding_function=get_embedder())
    collection.add(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"source": chunk["source"], "page": chunk["page"], "chunk_index": chunk["chunk_index"]}
            for chunk in chunks
        ],
    )

    print(f"[OK] stored {collection.count()} chunk(s) in collection '{name}'")
    return collection


# ============================================================
# RETRIEVE: question -> closest chunks   (Bootcamp retrieve())
# ============================================================
def search(collection, question, k=TOP_K, max_dist=MAX_DISTANCE):
    """Return up to k chunks closest in meaning to the question, closest first.

    If max_dist is set, chunks farther than max_dist are dropped
    (so the result can be an empty list -> the caller should refuse).
    max_dist=None means no filtering (threshold not chosen yet).
    """
    if not question.strip():
        raise ValueError("Question is empty.")
    if k <= 0:
        raise ValueError(f"k must be greater than 0 (got {k})")

    k = min(k, collection.count())     # cannot ask for more results than stored chunks

    result = collection.query(query_texts=[question], n_results=k)

    hits = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if max_dist is not None and distance > max_dist:
            continue

        hits.append({
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance,
        })

    return hits


# ============================================================
# Manual check:  python vector_store.py "your question"
# ============================================================
if __name__ == "__main__":
    import sys

    from document_loader import load_documents
    from text_chunker import chunk_pages

    collection = build_collection(chunk_pages(load_documents()))

    question = " ".join(sys.argv[1:]) or "What is this document about?"
    print(f"\nQuestion: {question}\n")
    for hit in search(collection, question):
        print(f"--- distance {hit['distance']:.3f} | {hit['source']} | page {hit['page']} "
              f"| chunk {hit['chunk_index']} ---")
        print(hit["text"][:150], "\n")
