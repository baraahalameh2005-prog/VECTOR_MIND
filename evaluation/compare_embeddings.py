"""
compare_embeddings.py — compare the OLD embedding model (Chroma default,
all-MiniLM-L6-v2) with the NEW one (multilingual-e5-small) on the same chunks
and the same pre-registered questions.

Usage (from backend/, venv active):
    python ../evaluation/compare_embeddings.py <folder with the PDF>

The question files (questions_gold.json, bilingual_gold.json) contain Gold chunk
ids written for the judging-guide PDF (IEEE_RAG_Hackathon_Judging_Guide_AR.pdf).
For other documents, write new question files first.

Each model gets its own collection - embeddings are never mixed.
Metrics: Hit@1/3/5 (is a correct chunk in the top k?), Recall@5 (multi-chunk
questions: share of correct chunks in the top 5), mean rank of the first correct
chunk, and separation (share of answerable/unanswerable question pairs where
the answerable one has the closer top-1 result).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from document_loader import load_documents
from embedder import get_embedder
from text_chunker import chunk_pages


def first_gold_ranks(collection, questions, n):
    rows = []
    for item in questions:
        result = collection.query(query_texts=[item["q"]], n_results=n)
        got = result["ids"][0]
        rows.append({**item, "ranks": sorted(got.index(g) + 1 for g in item["gold"]),
                     "top1_distance": result["distances"][0][0]})
    return rows


def summarize(rows):
    answerable = [r for r in rows if r["gold"]]
    hit = lambda k: sum(r["ranks"][0] <= k for r in answerable)
    line = (f"Hit@1 {hit(1)}/{len(answerable)} | Hit@3 {hit(3)}/{len(answerable)} | "
            f"Hit@5 {hit(5)}/{len(answerable)} | mean rank {sum(r['ranks'][0] for r in answerable) / len(answerable):.1f}")
    multi = [r for r in answerable if len(r["gold"]) > 1 and r.get("type") == "multi_chunk"]
    if multi:
        recall = sum(sum(x <= 5 for x in r["ranks"]) / len(r["gold"]) for r in multi) / len(multi)
        line += f" | Recall@5 (multi-chunk) {recall:.2f}"
    unanswerable = [r for r in rows if not r["gold"]]
    if unanswerable:
        pairs = [(a["top1_distance"], u["top1_distance"]) for a in answerable for u in unanswerable]
        line += f" | separation {sum(a < u for a, u in pairs) / len(pairs):.0%}"
    return line


def main(pdf_folder):
    chunks = chunk_pages(load_documents(pdf_folder))
    ids = [f"p{c['page']}c{c['chunk_index']}" for c in chunks]
    texts = [c["text"] for c in chunks]

    main_questions = json.load(open(HERE / "questions_gold.json", encoding="utf-8"))["questions"]
    bilingual = json.load(open(HERE / "bilingual_gold.json", encoding="utf-8"))
    english_docs = bilingual["synthetic_english_docs"]

    client = chromadb.Client()
    models = {"MiniLM (old, Chroma default)": DefaultEmbeddingFunction(),
              "E5 (new, multilingual-e5-small)": get_embedder()}

    for label, embedding_function in models.items():
        slug = "minilm" if "MiniLM" in label else "e5"
        pdf_col = client.create_collection(f"compare_{slug}_pdf", embedding_function=embedding_function)
        pdf_col.add(ids=ids, documents=texts)
        en_col = client.create_collection(f"compare_{slug}_en", embedding_function=embedding_function)
        en_col.add(ids=list(english_docs), documents=list(english_docs.values()))

        print(f"\n=== {label} ===")
        for name, col, questions, n in [
            ("Arabic questions, Arabic PDF (23)", pdf_col, main_questions, len(ids)),
            ("English questions, Arabic PDF", pdf_col, bilingual["case3_en_question_ar_pdf"], len(ids)),
            ("Mixed AR+EN questions, Arabic PDF", pdf_col, bilingual["case5_mixed_question_ar_pdf"], len(ids)),
            ("English questions, English docs*", en_col, bilingual["case2_en_question_en_docs"], len(english_docs)),
            ("Arabic questions, English docs*", en_col, bilingual["case4_ar_question_en_docs"], len(english_docs)),
        ]:
            print(f"{name:36s} {summarize(first_gold_ranks(col, questions, n))}")
    print("\n* English docs = 8 short hand-written passages (no real English PDF yet).")


if __name__ == "__main__":
    main(sys.argv[1])
