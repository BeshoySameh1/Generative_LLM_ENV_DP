"""
build_index.py
----------------
Run this ONCE (in Colab or locally) whenever you add/update PDFs.

Improvements over v1:
- Extracts TABLES properly (pdfplumber's extract_tables), turning each row
  into a clean, self-contained sentence — instead of letting table text get
  mangled by plain text extraction.
- Falls back to plain paragraph text for any non-table content on the page.
- Chunks tables per-row (one product = one chunk) so facts like price/type
  never get split away from the product they belong to.

Usage:
    python build_index.py

Outputs (commit both to your GitHub repo, same folder as app.py):
    - product_index.faiss
    - chunks.pkl
"""

import os
import re
import pickle

import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer

PDF_FOLDER = "pdfs"
INDEX_PATH = "product_index.faiss"
CHUNKS_PATH = "chunks.pkl"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Fallback chunking for plain paragraph text (non-table content)
TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 80


def clean_cell(value):
    """Normalize a table cell's text."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def row_to_sentence(header, row):
    """
    Turn a table row into a clean, self-contained natural-language sentence
    using the header labels, e.g.:
        "Name: Botox Metox | Type: Botulinum toxin | Price: 1200 EGP"
    This keeps every fact about one product together in a single chunk.
    """
    parts = []
    for col_name, cell_value in zip(header, row):
        col_name = clean_cell(col_name)
        cell_value = clean_cell(cell_value)
        if col_name and cell_value:
            parts.append(f"{col_name}: {cell_value}")
    return " | ".join(parts)


def extract_tables_as_chunks(pdf_path, source_name):
    """Extract every table in the PDF, one chunk per data row."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue  # need at least a header + one data row
                header = table[0]
                for row in table[1:]:
                    sentence = row_to_sentence(header, row)
                    if sentence:
                        chunks.append({
                            "text": sentence,
                            "source": source_name,
                            "page": page_num,
                            "type": "table_row",
                        })
    return chunks


def extract_plain_text_as_chunks(pdf_path, source_name):
    """
    Extract non-table paragraph text (e.g. descriptions, intro text, FAQ)
    using a sliding-window chunker, in case the PDF has prose sections too.
    """
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\n{2,}", "\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = start + TEXT_CHUNK_SIZE
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "source": source_name,
                        "page": page_num,
                        "type": "paragraph",
                    })
                start += TEXT_CHUNK_SIZE - TEXT_CHUNK_OVERLAP
    return chunks


def main():
    if not os.path.isdir(PDF_FOLDER):
        raise SystemExit(
            f"Folder '{PDF_FOLDER}/' not found. Create it and put your PDF files inside."
        )

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise SystemExit(f"No PDF files found inside '{PDF_FOLDER}/'.")

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    all_chunks = []
    for fname in pdf_files:
        path = os.path.join(PDF_FOLDER, fname)
        print(f"\nProcessing: {fname}")

        table_chunks = extract_tables_as_chunks(path, fname)
        print(f"  Table rows extracted: {len(table_chunks)}")

        text_chunks = extract_plain_text_as_chunks(path, fname)
        print(f"  Paragraph chunks extracted: {len(text_chunks)}")

        all_chunks.extend(table_chunks)
        all_chunks.extend(text_chunks)

    if not all_chunks:
        raise SystemExit("No content could be extracted. Are the PDFs scanned images?")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    print("Embedding chunks...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nSaved index to '{INDEX_PATH}'")
    print(f"Saved chunks to '{CHUNKS_PATH}'")

    # Quick sanity preview
    print("\n--- Sample chunks ---")
    for c in all_chunks[:5]:
        print(f"[{c['type']}] {c['text'][:150]}")

    print("\nDone. Commit product_index.faiss + chunks.pkl to your GitHub repo.")


if __name__ == "__main__":
    main()
