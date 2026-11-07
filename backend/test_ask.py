"""
test_ask.py — quick end-to-end check: build the index, ask 4 questions, print the results.

Run from backend/:   .venv\\Scripts\\python test_ask.py
"""

import sys
import time

from rag_pipeline import ask, build_index
from vector_store import search
import rag_pipeline

QUESTIONS = [
    ("كم ساعة بقدر أسجل بالفصل؟", "answered"),
    ("شو المتطلب السابق لمساق برمجة الويب 1؟", "answered"),
    ("متى امتحان مادة الخوارزميات؟", "refused"),
    ("اكتب لي قصيدة عن الجامعة", "refused"),
    ("كم ساعة تخصص علم الحاسوب؟", "refused"),            # standalone CS hours are not in the documents
    ("كم ساعة علم الحاسوب فرعي إدارة أعمال؟", "answered"),
]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    build_index()

    passed = 0
    for question, expected in QUESTIONS:
        print("=" * 70)
        print(f"Q: {question}   (expected: {expected})")
        for hit in search(rag_pipeline._collection, question):
            print(f"   retrieved: {hit['distance']:.3f} | {hit['source']} p{hit['page']}")
        start = time.time()
        result = ask(question)
        print(f"status: {result['status']}   ({time.time() - start:.1f}s)")
        print(f"answer: {result['answer']}")
        for source in result["sources"]:
            print(f"   source: {source['source']} | page {source['page']}")
        ok = result["status"] == expected
        passed += ok
        print("PASS" if ok else "FAIL")

    print("=" * 70)
    print(f"{passed}/{len(QUESTIONS)} passed")
