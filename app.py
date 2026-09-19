"""
app.py
-------
Streamlit RAG chatbot. Loads a prebuilt FAISS index + chunk store
(product_index.faiss, chunks.pkl — created by build_index.py) and answers
questions about your PDFs using retrieval + Flan-T5 generation.

Deploy: push this file + product_index.faiss + chunks.pkl + requirements.txt
to a GitHub repo, then deploy on https://share.streamlit.io (free).
"""

import pickle

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

INDEX_PATH = "product_index.faiss"
CHUNKS_PATH = "chunks.pkl"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEN_MODEL_NAME = "google/flan-t5-base"

TOP_K = 3                      # how many chunks to retrieve per question
MAX_CONTEXT_CHARS = 1800       # safety cap so the prompt doesn't overflow
MIN_SIMILARITY_DISTANCE = None # see note in retrieve() if you want a cutoff


@st.cache_resource(show_spinner="Loading models and index (first load can take a minute)...")
def load_everything():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(GEN_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(GEN_MODEL_NAME)

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return embedder, tokenizer, model, index, chunks


embedder, tokenizer, model, index, chunks = load_everything()


def retrieve(query, k=TOP_K):
    """Return the top-k most relevant chunk dicts for a query."""
    q_emb = embedder.encode([query])
    distances, indices = index.search(q_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append(chunks[idx])
    return results


def build_prompt(question, retrieved_chunks):
    context = "\n---\n".join(c["text"] for c in retrieved_chunks)
    context = context[:MAX_CONTEXT_CHARS]
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not contained in the context, say you don't have that information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt


def answer_question(question):
    retrieved = retrieve(question)
    if not retrieved:
        return "I couldn't find anything relevant in the documents.", []

    prompt = build_prompt(question, retrieved)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    sources = sorted(set(c["source"] for c in retrieved))
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
