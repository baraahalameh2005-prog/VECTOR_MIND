"""
document_loader.py — LOAD stage of the RAG pipeline.

Supports PDF files only. Reads every PDF in ORIGINAL_DOCS_DIR and returns
one record per page:
    {"text": "...", "source": "file.pdf", "page": 1}

Text is normalized with Unicode NFKC, which converts Arabic "presentation
forms" (glyph shapes stored by many PDFs) back into standard Arabic letters.

Each file gets one status line:
    [OK]          text extracted
    [NO TEXT]     file opened, but no page had extractable text (e.g. scanned) - skipped
    [UNREADABLE]  file could not be opened (corrupt / not a real PDF)          - skipped
Pages without text inside a good file are skipped with a [WARNING].
Missing folder, no PDFs, or no text from any file -> raises an error.

Known limitation (decision B1 - revisit when testing retrieval):
In some Arabic PDFs, the WORDS inside a line come out in reverse order
(letters inside each word are correct). We do not reorder words yet.
"""

import unicodedata
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import ORIGINAL_DOCS_DIR


# ============================================================
# One PDF file -> list of pages
# ============================================================
def load_pdf(pdf_path):
    """Extract the text of each page of one PDF file.

    Page numbers start at 1 (as a human reads them), not 0.
    Raises PdfReadError if the file cannot be opened.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = unicodedata.normalize("NFKC", raw_text).strip()

        if not text:
            print(f"[WARNING] {pdf_path.name} page {page_number}: "
                  "no extractable text (empty or scanned page) - skipped")
            continue

        pages.append({"text": text, "source": pdf_path.name, "page": page_number})

    return pages


# ============================================================
# Whole folder -> list of pages from all PDFs
# ============================================================
def load_documents(folder=ORIGINAL_DOCS_DIR):
    """Load every PDF in the folder and return all their pages in one list."""
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder}")

    pdf_paths = sorted(folder.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {folder}")

    all_pages = []
    for pdf_path in pdf_paths:
        try:
            pages = load_pdf(pdf_path)
        except PdfReadError as error:
            print(f"[UNREADABLE] {pdf_path.name}: cannot be opened ({error}) - skipped")
            continue

        if not pages:
            print(f"[NO TEXT] {pdf_path.name}: opened, but no page has extractable text "
                  "(possibly scanned - OCR is not supported) - skipped")
            continue

        print(f"[OK] {pdf_path.name}: {len(pages)} page(s) loaded")
        all_pages.extend(pages)

    if not all_pages:
        raise ValueError("No text could be extracted from any PDF file.")

    return all_pages


# ============================================================
# Manual check:  python document_loader.py
# ============================================================
if __name__ == "__main__":
    loaded_pages = load_documents()
    print(f"\nTotal pages loaded: {len(loaded_pages)}\n")
    for record in loaded_pages:
        print(f"--- {record['source']} | page {record['page']} | {len(record['text'])} chars ---")
        print(record["text"][:200], "\n")
