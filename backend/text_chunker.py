"""
text_chunker.py — CHUNK stage of the RAG pipeline.

Input  (from document_loader.load_documents):
    [{"text": "...", "source": "file.pdf", "page": 1}, ...]

Output (one record per chunk):
    [{"text": "...", "source": "file.pdf", "page": 1, "chunk_index": 1}, ...]

Chunking is character-based, using CHUNK_SIZE and CHUNK_OVERLAP from config.py
(Bootcamp method: step = size - overlap).

Each page is chunked separately, so a chunk never crosses a page boundary and
always carries a correct source + page number.
Known limitation: an idea that starts at the end of one page and continues on
the next page is split between two chunks (to be measured in evaluation).
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP


# ============================================================
# One text -> list of overlapping pieces   (Bootcamp chunk())
# ============================================================
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into pieces of at most `size` characters.

    Neighbouring pieces share exactly `overlap` characters.
    Stops as soon as a piece reaches the end of the text, so no extra
    piece is created that only repeats the end of the previous one.
    """
    if size <= 0:
        raise ValueError(f"size must be greater than 0 (got {size})")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be between 0 and size-1 (got overlap={overlap}, size={size})")

    chunks = []
    step = size - overlap

    for start in range(0, len(text), step):
        piece = text[start:start + size]
        chunks.append(piece)

        if start + size >= len(text):   # this piece already reached the end
            break

    return chunks


# ============================================================
# Pages -> chunks with metadata
# ============================================================
def chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk every page and copy its source + page number into each chunk.

    chunk_index counts the chunks inside one page, starting at 1.
    """
    all_chunks = []

    for page in pages:
        pieces = chunk_text(page["text"], size, overlap)

        for chunk_index, piece in enumerate(pieces, start=1):
            all_chunks.append({
                "text": piece,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": chunk_index,
            })

    return all_chunks


# ============================================================
# Manual check:  python text_chunker.py
# ============================================================
if __name__ == "__main__":
    from document_loader import load_documents

    loaded_pages = load_documents()
    chunks = chunk_pages(loaded_pages)
    print(f"\n{len(loaded_pages)} page(s) -> {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")
    for record in chunks:
        print(f"--- {record['source']} | page {record['page']} | chunk {record['chunk_index']} "
              f"| {len(record['text'])} chars ---")
        print(record["text"][:150], "\n")
