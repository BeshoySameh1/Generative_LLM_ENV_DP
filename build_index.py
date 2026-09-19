"""
build_index.py
----------------
Run this ONCE (locally or in a notebook) whenever you add/update PDFs.
It reads every PDF inside the pdfs/ folder, extracts text, splits it into
overlapping chunks, embeds them with MiniLM, and saves:
    - product_index.faiss   (the vector index)
    - chunks.pkl            (the raw text chunks, aligned with the index)

Usage:
    python build_index.py

Then commit product_index.faiss + chunks.pkl to your GitHub repo alongside
app.py so the deployed app can load them directly (no PDFs needed at runtime).
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

CHUNK_SIZE = 500       # characters per chunk (roughly ~100-120 tokens)
CHUNK_OVERLAP = 80     # overlap between consecutive chunks


def extract_text_from_pdf(path):
    """Extract raw text from a single PDF file, page by page."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text


def clean_text(text):
    """Collapse excessive whitespace/newlines."""
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Simple sliding-window chunker over characters.
    Each chunk stores which source PDF it came from, so answers can be
    traced back later if needed.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source_name})
        start += chunk_size - overlap
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
        print(f"  Extracting: {fname}")
        raw_text = extract_text_from_pdf(path)
        cleaned = clean_text(raw_text)
        file_chunks = chunk_text(cleaned, source_name=fname)
        print(f"    -> {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    if not all_chunks:
        raise SystemExit("No text could be extracted from the PDFs. Are they scanned images?")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    print("Embedding chunks...")
    embeddings = embedder.encode(
        texts, convert_to_numpy=True, show_progress_bar=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nSaved index to '{INDEX_PATH}'")
    print(f"Saved chunks to '{CHUNKS_PATH}'")
    print("\nDone. Commit both files to your GitHub repo along with app.py.")


if __name__ == "__main__":
    main()
