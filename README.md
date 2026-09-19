# Product Assistant — RAG Chatbot (v2: table-aware + hybrid answering)

## What changed from v1

1. **Table extraction** — `build_index.py` now uses `pdfplumber`'s
   `extract_tables()` instead of plain text extraction. Each table row
   (e.g. one product) becomes ONE clean chunk like:
   `"Name: Botox Metox | Type: Botulinum toxin | Price: 1200 EGP"`
   This stops facts from getting split apart or garbled, which was the
   root cause of vague/wrong answers in v1.

2. **Hybrid answering in `app.py`**:
   - First tries an **extractive QA model** (`deepset/roberta-base-squad2`)
     which pulls the literal answer span from the retrieved text — ideal
     for "what is the price / type / function of X" questions.
   - Falls back to **Flan-T5-large** (upgraded from `base`) for broader or
     open-ended questions where extractive QA isn't confident.

3. **Retrieval** — `TOP_K` raised from 3 to 5 chunks per question, since
   chunks are now smaller and more precise (one product per chunk).

## Folder structure

```
rag-chatbot/
├── pdfs/                  <- your product PDF files
├── build_index.py         <- run once (or after PDF updates) to build the index
├── app.py                 <- the Streamlit chatbot app
├── requirements.txt
├── product_index.faiss    <- generated (commit this)
├── chunks.pkl             <- generated (commit this)
└── README.md
```

## How to rebuild the index (Colab, same as before)

```python
!git clone https://github.com/yourusername/your-repo-name.git
%cd your-repo-name
!pip install pdfplumber faiss-cpu sentence-transformers
!python build_index.py
```

Download the two generated files (`product_index.faiss`, `chunks.pkl`) and
upload them to your GitHub repo, replacing the old ones.

## Sanity-check your extraction BEFORE rebuilding the full index

If answers are still off after this update, first check what's actually
being extracted from your PDF's tables:

```python
import pdfplumber

with pdfplumber.open("pdfs/products.pdf") as pdf:
    for page in pdf.pages:
        for table in page.extract_tables():
            for row in table:
                print(row)
```

If this prints `None` or garbled cells, your PDF's tables aren't real
tables (they might be images or manually spaced text) — pdfplumber can't
parse those. In that case, either:
- re-export the PDF from its original source (Word/Excel) with real table
  formatting rather than a scanned/flattened version, or
- manually convert your product list into a plain structured text/CSV file
  and feed that in as "plain text" chunks instead.

## Deploy (same as before)

1. Push the updated `app.py`, `build_index.py`, `requirements.txt`,
   `product_index.faiss`, and `chunks.pkl` to your GitHub repo.
2. Streamlit Cloud auto-redeploys on push if already connected, or set up
   fresh at https://share.streamlit.io (repo + branch + `app.py` as entry).
3. First load will be slower this time — Flan-T5-large and the QA model
   are both bigger downloads than v1's setup. Expect the free tier to feel
   a bit heavier; if it's too slow/crashes on memory, drop back to
   `google/flan-t5-base` in `GEN_MODEL_NAME` inside `app.py`.

## Tuning knobs in `app.py`

- `EXTRACTIVE_CONFIDENCE_THRESHOLD` — lower it if the QA model is being
  too conservative and falling back to generative too often; raise it if
  it's returning low-quality extractive answers too confidently.
- `TOP_K` — more chunks = more context but slower + more chance of noise.
- `GEN_MODEL_NAME` — swap to `flan-t5-base` if `flan-t5-large` is too slow
  or too much memory for your deployment tier.
