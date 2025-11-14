import streamlit as st
import fitz
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

try:
    import faiss
except:
    faiss = None

try:
    from huggingface_hub import InferenceClient
except:
    InferenceClient = None


# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="SmartRAG – Advanced RAG Chatbot", page_icon="🤖", layout="wide")
st.markdown("<style>header{visibility:hidden;}</style>", unsafe_allow_html=True)
st.title("🤖 SmartRAG — Advanced RAG Chatbot")
st.caption("Self-RAG • HyDE • RAG-Fusion • Hybrid Retrieval")

# ============================
# SIDEBAR
# ============================
st.sidebar.title("⚙️ Settings")

MODEL_CHOICE = st.sidebar.selectbox(
    "Choose LLM",
    [
        "microsoft/phi-3-mini-4k-instruct",
        "google/gemma-2-2b-it",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ], index=0
)

RETRIEVAL_METHOD = st.sidebar.selectbox(
    "Retrieval Mode",
    ["Standard RAG", "HyDE", "RAG-Fusion (Query Expansion)", "Self-RAG"]
)

top_k = st.sidebar.slider("Top-K Chunks", 2, 8, 4)

# ============================
# HF TOKEN & CLIENT
# ============================
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

@st.cache_resource
def get_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_hf_client(token):
    if token is None:
        return None
    try:
        return InferenceClient(token=token)
    except Exception as e:
        st.error("HF Client error: " + str(e))
        return None

embed_model = get_embed_model()
hf_client = get_hf_client(HF_TOKEN)


# ============================
# PDF TEXT EXTRACTION
# ============================
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])


def chunk_text(text, max_words=200):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, wc = [], [], 0

    for s in sentences:
        words = s.split()
        if wc + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur, wc = [s], len(words)
        else:
            cur.append(s)
            wc += len(words)

    if cur:
        chunks.append(" ".join(cur))

    return [c.strip() for c in chunks if c.strip()]


# ============================
# INDEX CREATION
# ============================
@st.cache_data(ttl=3600)
def build_index(chunks):
    embeddings = embed_model.encode(chunks, normalize_embeddings=True).astype("float32")

    if faiss:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        index = None

    bm25 = BM25Okapi([c.split() for c in chunks])
    return index, bm25, embeddings


# ============================
# RETRIEVAL FUNCTIONS
# ============================
def retrieve_standard(query, index, bm25, chunks, embeddings=None):
    qv = embed_model.encode([query], normalize_embeddings=True).astype("float32")

    if index:
        _, idx = index.search(qv, top_k)
        return [chunks[i] for i in idx[0]]

    sims = np.dot(embeddings, qv[0])
    ids = np.argsort(-sims)[:top_k]
    return [chunks[i] for i in ids]


def chat(model, messages, max_tokens=200, temperature=0.2):
    """Wrapper for HF chat-completion"""
    try:
        resp = hf_client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Model error: {e}"


def retrieve_hyde(query, index, bm25, chunks, embeddings=None):
    if hf_client is None:
        return retrieve_standard(query, index, bm25, chunks, embeddings)

    hypothetic = chat(
        MODEL_CHOICE,
        [{"role": "user", "content": f"Write a short factual answer:\n{query}"}],
        max_tokens=80
    )

    qv = embed_model.encode([hypothetic], normalize_embeddings=True).astype("float32")

    if index:
        _, idx = index.search(qv, top_k)
        return [chunks[i] for i in idx[0]]

    sims = np.dot(embeddings, qv[0])
    ids = np.argsort(-sims)[:top_k]
    return [chunks[i] for i in ids]


def generate_subqueries(q):
    if hf_client is None:
        parts = q.split()
        return [q] + parts[:3]

    txt = chat(
        MODEL_CHOICE,
        [{"role": "user", "content": f"Generate 3 short search queries:\n{q}"}],
        max_tokens=60
    )

    return [line.strip() for line in txt.split("\n") if line.strip()][:4]


def rrf(scores):
    final = {}
    for q, docs in scores.items():
        for rank, idx in enumerate(docs):
            final[idx] = final.get(idx, 0) + 1 / (60 + rank + 1)
    return sorted(final.items(), key=lambda x: x[1], reverse=True)


def retrieve_ragfusion(query, index, bm25, chunks, embeddings=None):
    sub_qs = generate_subqueries(query)
    scores = {}

    for sq in sub_qs:
        qv = embed_model.encode([sq], normalize_embeddings=True).astype("float32")
        if index:
            _, idx = index.search(qv, top_k)
            scores[sq] = idx[0]
        else:
            sims = np.dot(embeddings, qv[0])
            ids = np.argsort(-sims)[:top_k]
            scores[sq] = ids

    fused = rrf(scores)
    ids = [i for i, _ in fused[:top_k]]
    return [chunks[i] for i in ids]


# ============================
# FINAL ANSWER GENERATION
# ============================
def generate_answer(context, query):
    if hf_client is None:
        if query.lower() in context.lower():
            return context
        return "Context does not contain the answer. (HF token missing)"

    messages = [
        {"role": "system", "content": "Answer using ONLY the provided context."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nFinal Answer:"}
    ]

    return chat(MODEL_CHOICE, messages, max_tokens=250, temperature=0.2)


def self_critique(answer, context, query):
    if hf_client is None:
        return answer

    critique = chat(
        MODEL_CHOICE,
        [{"role": "user",
          "content": f"Critique this answer.\nQUESTION:{query}\nCONTEXT:{context}\nANSWER:{answer}\n"}],
        max_tokens=120,
        temperature=0
    )

    if "OK" in critique.upper():
        return answer

    improved = chat(
        MODEL_CHOICE,
        [{"role": "user",
          "content": f"Improve the answer.\nCRITIQUE:{critique}\nANSWER:{answer}"}],
        max_tokens=200
    )

    return improved


# ============================
# MAIN UI
# ============================
uploaded_pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])
query = st.text_input("Ask a question:")

if uploaded_pdf:
    with st.spinner("Processing PDF..."):
        text = extract_text(uploaded_pdf)
        chunks = chunk_text(text)
        index, bm25, embeddings = build_index(chunks)

    if query:
        if RETRIEVAL_METHOD == "Standard RAG":
            selected = retrieve_standard(query, index, bm25, chunks, embeddings)

        elif RETRIEVAL_METHOD == "HyDE":
            selected = retrieve_hyde(query, index, bm25, chunks, embeddings)

        elif RETRIEVAL_METHOD == "RAG-Fusion (Query Expansion)":
            selected = retrieve_ragfusion(query, index, bm25, chunks, embeddings)

        elif RETRIEVAL_METHOD == "Self-RAG":
            selected = retrieve_standard(query, index, bm25, chunks, embeddings)

        context = "\n\n".join(selected)
        answer = generate_answer(context, query)

        if RETRIEVAL_METHOD == "Self-RAG":
            answer = self_critique(answer, context, query)

        st.subheader("🧠 Final Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Chunks"):
            for c in selected:
                st.write(c)

else:
    st.info("Upload a PDF to start.")

st.caption("SmartRAG — Hybrid RAG pipeline")
