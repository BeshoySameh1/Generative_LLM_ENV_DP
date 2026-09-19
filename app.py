"""
app.py
-------
Streamlit RAG chatbot — hybrid answering:
  1. Retrieve top-k relevant chunks (MiniLM + FAISS)
  2. Run an EXTRACTIVE QA model to pull the precise fact (price, type, etc.)
     directly from the retrieved text — no hallucination, no paraphrasing.
  3. Use Flan-T5 (generative) to phrase a natural fallback answer when the
     extractive model isn't confident, or for broader/open-ended questions.

This fixes the "vague non-answer" problem from pure generative-only setups.

Requires: product_index.faiss + chunks.pkl (created by build_index.py) in
the same folder as this file.
"""

import pickle

import torch
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)

INDEX_PATH = "product_index.faiss"
CHUNKS_PATH = "chunks.pkl"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEN_MODEL_NAME = "google/flan-t5-base"         # large caused OOM kills on free-tier hosting
QA_MODEL_NAME = "deepset/roberta-base-squad2"  # extractive QA

TOP_K = 5                     # retrieve more chunks now that they're smaller/cleaner
MAX_CONTEXT_CHARS = 2000
EXTRACTIVE_CONFIDENCE_THRESHOLD = 0.15  # below this, fall back to generative


@st.cache_resource(show_spinner="Loading models and index (first load can take a minute)...")
def load_everything():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    gen_tokenizer = T5Tokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = T5ForConditionalGeneration.from_pretrained(GEN_MODEL_NAME)

    # transformers v5 removed the pipeline("question-answering", ...) shortcut,
    # so we load the QA model/tokenizer directly and extract the answer span ourselves.
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
    qa_model.eval()

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return embedder, gen_tokenizer, gen_model, qa_tokenizer, qa_model, index, chunks


embedder, gen_tokenizer, gen_model, qa_tokenizer, qa_model, index, chunks = load_everything()


def retrieve(query, k=TOP_K):
    """Return the top-k most relevant chunk dicts for a query."""
    q_emb = embedder.encode([query])
    distances, indices = index.search(q_emb, k)
    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(chunks[idx])
    return results


def build_context(retrieved_chunks, max_chars=MAX_CONTEXT_CHARS):
    context = "\n".join(c["text"] for c in retrieved_chunks)
    return context[:max_chars]


def extractive_answer(question, context):
    """
    Manual replacement for the (removed-in-v5) pipeline("question-answering", ...).
    Runs the QA model directly and extracts the highest-scoring answer span.
    Returns (answer_text, confidence_score).
    """
    inputs = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start_idx = int(torch.argmax(start_logits))
    end_idx = int(torch.argmax(end_logits))

    if end_idx < start_idx:
        return "", 0.0

    input_ids = inputs["input_ids"][0]
    answer_tokens = input_ids[start_idx:end_idx + 1]
    answer = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    # Rough confidence: combined softmax probability of the chosen start/end positions
    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)
    score = float(start_probs[start_idx] * end_probs[end_idx])

    return answer, score


def generative_answer(question, context):
    prompt = (
        "You are a precise product-information assistant. "
        "Answer the question in one direct sentence using ONLY facts from the context. "
        "If the exact answer is not explicitly stated in the context, respond exactly: "
        "\"That information isn't in the documents.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    outputs = gen_model.generate(**inputs, max_new_tokens=150)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)


def answer_question(question):
    retrieved = retrieve(question)
    if not retrieved:
        return "I couldn't find anything relevant in the documents.", []

    context = build_context(retrieved)
    sources = sorted(set(c["source"] for c in retrieved))

    # Step 1: try extractive QA first — best for precise facts (price, type, etc.)
    try:
        ext_answer, ext_score = extractive_answer(question, context)
    except Exception:
        ext_answer, ext_score = "", 0.0

    if ext_score >= EXTRACTIVE_CONFIDENCE_THRESHOLD and ext_answer:
        return ext_answer, sources

    # Step 2: fall back to generative answer for broader/open-ended questions
    answer = generative_answer(question, context)
    return answer, sources


# ---------------- UI ----------------

st.set_page_config(page_title="Product Assistant", page_icon="💬")
st.title("💬 Product Assistant")
st.caption("Ask a question about our products — answers are grounded in our product documents.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

question = st.chat_input("Ask about our products...")
if question:
    st.session_state.history.append(("user", question))
    st.chat_message("user").write(question)

    with st.spinner("Thinking..."):
        reply, sources = answer_question(question)

    if sources:
        reply_display = reply + f"\n\n*Source(s): {', '.join(sources)}*"
    else:
        reply_display = reply

    st.session_state.history.append(("assistant", reply_display))
    st.chat_message("assistant").write(reply_display)
