"""
rag_pipeline.py — connects the stages: build_index() and ask(question) -> {status, answer, sources}.

build_index()                -> load PDFs, chunk, embed and store in Chroma (once per run)
ask(question, history=None)  -> {"status": "answered" | "refused",
                                 "answer": "...",
                                 "sources": [{"source": "file.pdf", "page": 3}, ...]}

Refusal rules:
- search returns nothing          -> refuse WITHOUT calling the model
- the model writes REFUSAL_TEXT   -> refuse, no sources
- the answer cites no chunk [n], or is only citations with no text -> refuse
Sources come from the metadata of the chunks the model cited as [n].
Follow-up questions: the previous question is added to the search text, and the
last 2 turns are sent to the model (see generator.py).
"""

import re

from config import REFUSAL_TEXT
from document_loader import load_documents
from generator import generate
from text_chunker import chunk_pages
from vector_store import build_collection, search

_collection = None


def build_index():
    global _collection
    _collection = build_collection(chunk_pages(load_documents()))
    return _collection


def _refused():
    return {"status": "refused", "answer": REFUSAL_TEXT, "sources": []}


def _cited_sources(answer, hits):
    """Unique {source, page} for every [n] the model cited, in citation order."""
    sources, seen = [], set()
    for number in re.findall(r"\[(\d+)\]", answer):
        index = int(number) - 1
        if 0 <= index < len(hits):
            key = (hits[index]["source"], hits[index]["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"source": key[0], "page": key[1]})
    return sources


def ask(question, history=None):
    if _collection is None:
        build_index()
    if not question.strip():
        return _refused()

    search_text = question
    if history:
        search_text = f"{history[-1]['question']} {question}"

    hits = search(_collection, search_text)
    if not hits:
        return _refused()

    answer = generate(question, hits, history)
    if REFUSAL_TEXT.rstrip(".") in answer:
        return _refused()

    # Grounding guards: an answer must cite at least one retrieved chunk,
    # and must contain real text besides the [n] citations.
    sources = _cited_sources(answer, hits)
    content = re.sub(r"\[\d+\]", "", answer).strip(" .،\n")
    if not sources or len(content) < 5:
        return _refused()

    return {"status": "answered", "answer": answer, "sources": sources}
