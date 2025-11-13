import os
import re
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st
import numpy as np
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import faiss
from PyPDF2 import PdfReader

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================
st.set_page_config(page_title="RAGNova Fusion", page_icon="🤖", layout="centered")
st.title("🤖 RAGNova — Fusion Retrieval + Generation")

APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
DEFAULT_EMBED = "sentence-transformers/sentence-t5-large"
DEFAULT_LLM = "microsoft/Phi-3-mini-4k-instruct"


# =============================================================================
# SIDEBAR SETTINGS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Model & Settings")

    model_choice = st.selectbox(
        "Choose LLM:",
        [
            "microsoft/Phi-3-mini-4k-instruct",
            "tiiuae/Falcon3-3B-Instruct",
            "meta-llama/Llama-3.2-3b-instruct",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ],
        index=0
    )

    embed_model = st.selectbox(
        "Embedding Model",
        [DEFAULT_EMBED, "sentence-transformers/all-MiniLM-L12-v2"],
        index=0
    )

    top_k = st.slider("Chunks to retrieve:", 1, 10, 5)
    bm25_weight = st.slider("BM25 weight:", 0.0, 1.0, 0.6)
    dense_weight = st.slider("Dense weight:", 0.0, 1.0, 0.4)
    max_context_chars = st.slider("Max context characters:", 1000, 20000, 6000)
    max_gen_tokens = st.slider("Max new tokens:", 50, 600, 250)
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.3)

    if not HF_TOKEN:
        st.warning("⚠️ Add HF_TOKEN in Streamlit Secrets for Llama / Falcon models.")


# =============================================================================
# EMBEDDINGS (NO CACHE – avoids UnhashableParamError)
# =============================================================================
def get_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})


def get_hf_client(model_name: str):
    return InferenceClient(model=model_name, token=HF_TOKEN or None)


emb = get_embeddings(embed_model)
client = get_hf_client(model_choice)


# =============================================================================
# FAISS HELPERS
# =============================================================================
def faiss_paths() -> Tuple[Path, Path]:
    return FAISS_DIR / "index.faiss", FAISS_DIR / "index_meta.pkl"


def faiss_exists():
    idx, meta = faiss_paths()
    return idx.exists() and meta.exists()


def load_faiss_index():
    """Load FAISS + metadata safely."""
    if not faiss_exists():
        return None, None, None

    # Load LangChain FAISS wrapper
    db = FAISS.load_local(
        folder_path=str(FAISS_DIR),
        embeddings=emb,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

    # Load metadata text
    meta_path = FAISS_DIR / "index_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    texts = [m["text"] for m in meta]

    return db, BM25Okapi([t.split() for t in texts]), texts


# Load FAISS or fallback
db, bm25, stored_texts = load_faiss_index()

if db is None:
    st.warning("⚠️ No FAISS index found. Using small demo index.")

    demo_texts = [
        "Fusion RAG merges results from different queries.",
        "Reciprocal Rank Fusion improves retrieval robustness.",
        "Upload a PDF to build your real FAISS index."
    ]

    db = FAISS.from_texts(demo_texts, emb)
    stored_texts = demo_texts
    bm25 = BM25Okapi([t.split() for t in demo_texts])

retriever = db.as_retriever(search_kwargs={"k": top_k * 3})


# =============================================================================
# RAG FUNCTIONS
# =============================================================================
def expand_queries(q: str) -> List[str]:
    return [
        q,
        f"Explain {q}",
        f"Benefits of {q}",
        f"Challenges of {q}",
        f"Real-world applications of {q}"
    ]


def fuse_results(results: Dict[str, List]):
    """Perform RRF fusion."""
    scores = {}
    for q, docs in results.items():
        for r, d in enumerate(docs):
            txt = d.page_content if hasattr(d, "page_content") else str(d)
            scores[txt] = scores.get(txt, 0) + 1 / (60 + r + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def trim_context(text: str, max_len: int):
    return text if len(text) <= max_len else text[:max_len]


def build_prompt(context: str, question: str):
    return f"""
You are a helpful study assistant.
Use ONLY the provided context to answer.
If context lacks information: reply exactly:
"The context does not provide this information."

CONTEXT:
{context}

QUESTION: {question}

Answer clearly in 3–6 sentences:
"""


# =============================================================================
# PDF INDEX BUILDER
# =============================================================================
def build_index_from_pdf(file):
    reader = PdfReader(file)
    text = "\n\n".join([p.extract_text() or "" for p in reader.pages])

    chunks, size, overlap = [], 900, 150
    i, N = 0, len(text)
    while i < N:
        j = min(i + size, N)
        chunks.append(text[i:j])
        i = j - overlap if j < N else j

    # embed
    vecs = emb.embed_documents(chunks)
    arr = np.asarray(vecs, dtype="float32")

    # normalize
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)

    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    # save
    FAISS_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))

    meta = [{"text": chunks[i], "source": f"pdf#{i}"} for i in range(len(chunks))]
    with open(FAISS_DIR / "index_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # LC wrapper
    db2 = FAISS.from_texts(chunks, emb)
    db2.save_local(str(FAISS_DIR), index_name=INDEX_NAME)


# =============================================================================
# PDF UPLOAD UI
# =============================================================================
st.subheader("📘 Build / Load FAISS Index")

uploaded_pdf = st.file_uploader("Upload PDF to create index:")

if st.button("Build FAISS Index"):
    if uploaded_pdf:
        with st.spinner("Building FAISS index..."):
            build_index_from_pdf(uploaded_pdf)
        st.success("Index built successfully! Reload page.")
        st.experimental_rerun()
    else:
        st.warning("Upload a PDF first.")


# =============================================================================
# QUERY INTERFACE
# =============================================================================
st.subheader("💬 Ask a question")

query = st.text_input("Your question:")

if "chat" not in st.session_state:
    st.session_state.chat = []

if query:
    # retrieve
    expanded = expand_queries(query)
    results = {q: retriever.get_relevant_documents(q) for q in expanded}

    fused = fuse_results(results)
    top_chunks = [t for t, _ in fused][:top_k]

    context = trim_context("\n\n".join(top_chunks), max_context_chars)

    # LLM
    prompt = build_prompt(context, query)

    with st.spinner("Generating answer..."):
        response = client.text_generation(
            prompt,
            max_new_tokens=max_gen_tokens,
            temperature=temperature,
            stream=False
        )

    answer = (
        response.get("generated_text")
        if isinstance(response, dict)
        else str(response)
    )

    st.session_state.chat.append((query, answer, top_chunks))

# history
st.subheader("🧠 Conversation History")
for q, a, ctx in reversed(st.session_state.chat):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    with st.expander("Show retrieved context"):
        for c in ctx:
            st.write(c[:1000] + ("..." if len(c) > 1000 else ""))
    st.markdown("---")
