import streamlit as st
import fitz
import numpy as np
import faiss
import os
import torch
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="SmartRAG – Advanced AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 SmartRAG — Advanced RAG Chatbot (Free Deployment)")
st.caption("Self-RAG • HyDE • RAG-Fusion • Hybrid Retrieval • Free LLM API")


# =========================================================
# SIDEBAR SETTINGS
# =========================================================
st.sidebar.title("⚙️ Settings")

MODEL_CHOICE = st.sidebar.selectbox(
    "Choose LLM",
    [
        "google/gemma-2-2b-it",
        "mistral-7b-instruct-v0.3",
        "microsoft/phi-3-mini-4k-instruct",
        "meta-llama/Llama-3.1-8b-instruct"
    ]
)

RETRIEVAL_METHOD = st.sidebar.selectbox(
    "Retrieval Mode",
    ["Standard RAG", "HyDE", "RAG-Fusion (Query Expansion)", "Self-RAG"]
)

top_k = st.sidebar.slider("Top-K Chunks", 2, 8, 4)


# =========================================================
# GLOBALS
# =========================================================
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# PDF PROCESSING
# =========================================================
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, max_words=200):
    sentences = re.split(r"\. |\? |\!", text)
    chunks, cur, wc = [], [], 0
    for s in sentences:
        words = s.split()
        if wc + len(words) > max_words:
            chunks.append(" ".join(cur))
            cur, wc = [s], len(words)
        else:
            cur.append(s)
            wc += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# =========================================================
# BUILD HYBRID INDEX (FAISS + BM25)
# =========================================================
def build_index(chunks):
    embeddings = embed_model.encode(chunks, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    bm25 = BM25Okapi([c.split() for c in chunks])
    return index, bm25


# =========================================================
# RETRIEVAL MODES
# =========================================================

# ---- Standard RAG ----
def retrieve_standard(query, index, bm25, chunks):
    qv = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    _, idx = index.search(qv, top_k)
    return [chunks[i] for i in idx[0]]


# ---- HyDE: Hypothetical Answer ----
def retrieve_hyde(query, index, bm25, chunks):
    hypothetic = client.text_generation(
        model=MODEL_CHOICE,
        prompt=f"Write a short answer for: {query}",
        max_new_tokens=80
    )["generated_text"]

    qv = embed_model.encode([hypothetic], normalize_embeddings=True).astype("float32")
    _, idx = index.search(qv, top_k)
    return [chunks[i] for i in idx[0]]


# ---- RAG-Fusion (Query Expansion) ----
def generate_subqueries(q):
    prompt = f"""
Generate 4 search queries related to:
{q}
Output only the queries, one per line.
"""
    txt = client.text_generation(
        model=MODEL_CHOICE,
        prompt=prompt,
        max_new_tokens=100
    )["generated_text"]

    return [line.strip() for line in txt.split("\n") if line.strip()]


def rrf(scores, k=60):
    final = {}
    for q, docs in scores.items():
        for rank, idx in enumerate(docs):
            final[idx] = final.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(final.items(), key=lambda x: x[1], reverse=True)


def retrieve_ragfusion(query, index, bm25, chunks):
    sub_q = generate_subqueries(query)
    score_dict = {}

    for sq in sub_q:
        qv = embed_model.encode([sq], normalize_embeddings=True).astype("float32")
        _, idx = index.search(qv, top_k)
        score_dict[sq] = idx[0]

    fused = rrf(score_dict)
    top_ids = [i for i, _ in fused[:top_k]]
    return [chunks[i] for i in top_ids]


# ---- Self-RAG (Generation → Critique → Improve) ----
def self_critique(initial_answer, context, query):
    critique_prompt = f"""
You are a critic AI. Evaluate the answer.

QUESTION: {query}

CONTEXT:
{context}

ANSWER:
{initial_answer}

Critique the answer and list mistakes. If correct, write "OK".
"""
    critique = client.text_generation(
        model=MODEL_CHOICE,
        prompt=critique_prompt,
        max_new_tokens=200
    )["generated_text"]

    improve_prompt = f"""
Improve the answer based on this critique:

CRITIQUE: {critique}

Original Answer: {initial_answer}

Write an improved final answer.
"""
    improved = client.text_generation(
        model=MODEL_CHOICE,
        prompt=improve_prompt,
        max_new_tokens=250
    )["generated_text"]

    return improved


def retrieve_selfrag(query, index, bm25, chunks):
    return retrieve_standard(query, index, bm25, chunks)


# =========================================================
# FINAL ANSWER GENERATION
# =========================================================
def generate_answer(context, query):
    prompt = f"""
Use ONLY the context below.

If answer not found, say:
"The context does not provide this information."

CONTEXT:
{context}

QUESTION: {query}

Final Answer:
"""
    resp = client.text_generation(
        model=MODEL_CHOICE,
        prompt=prompt,
        max_new_tokens=250,
        temperature=0.2
    )["generated_text"]

    return resp.split("Final Answer:")[-1].strip()


# =========================================================
# MAIN UI
# =========================================================
uploaded_pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])
query = st.text_input("Ask a question:")

if uploaded_pdf:
    text = extract_text(uploaded_pdf)
    chunks = chunk_text(text)
    index, bm25 = build_index(chunks)

    if query:
        if RETRIEVAL_METHOD == "Standard RAG":
            selected = retrieve_standard(query, index, bm25, chunks)

        elif RETRIEVAL_METHOD == "HyDE":
            selected = retrieve_hyde(query, index, bm25, chunks)

        elif RETRIEVAL_METHOD == "RAG-Fusion (Query Expansion)":
            selected = retrieve_ragfusion(query, index, bm25, chunks)

        elif RETRIEVAL_METHOD == "Self-RAG":
            selected = retrieve_selfrag(query, index, bm25, chunks)

        context = "\n\n".join(selected)

        answer = generate_answer(context, query)

        if RETRIEVAL_METHOD == "Self-RAG":
            answer = self_critique(answer, context, query)

        st.subheader("🧠 Final Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Chunks"):
            for c in selected:
                st.write("- " + c)
else:
    st.info("Upload a PDF to begin.")
