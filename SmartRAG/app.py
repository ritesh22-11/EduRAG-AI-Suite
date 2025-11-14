import streamlit as st
import fitz
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

try:
    import faiss
except Exception as e:
    faiss = None


try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# STREAMLIT CONFIG
st.set_page_config(page_title="SmartRAG – Advanced RAG Chatbot (Free Deployment)", page_icon="🤖", layout="wide")
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
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/phi-3-mini-4k-instruct",
        "google/gemma-2-2b-it",
    ], index=0,
)

RETRIEVAL_METHOD = st.sidebar.selectbox(
    "Retrieval Mode",
    ["Standard RAG", "HyDE", "RAG-Fusion (Query Expansion)", "Self-RAG"],
)

top_k = st.sidebar.slider("Top-K Chunks", 2, 8, 4)

# =========================================================
# SECRETS & HF CLIENT
# =========================================================
# Use Streamlit secrets
HF_TOKEN = None
if "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]
else:
    # fallback to environment variable if available
    HF_TOKEN = os.getenv("HF_TOKEN")


@st.cache_resource
def get_embed_model():
    # cached to avoid reloading on each rerun
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_hf_client(token: str):
    if not token:
        return None
    if InferenceClient is None:
        st.warning("huggingface_hub not installed; model calls disabled.")
        return None
    try:
        return InferenceClient(token=token)
    except Exception as e:
        st.error("Failed to create HF Inference client: " + str(e))
        return None


embed_model = get_embed_model()
hf_client = get_hf_client(HF_TOKEN)

# =========================================================
# PDF PROCESSING
# =========================================================

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    return "\n".join(text_parts)



def chunk_text(text, max_words=200):
    # safer sentence split and chunking
    sentences = re.split(r"(?<=[\.!?])\s+", text)
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


# =========================================================
# BUILD HYBRID INDEX (FAISS + BM25)
# =========================================================
@st.cache_data(ttl=60 * 60)
def build_index(chunks):
    # embeddings
    embeddings = embed_model.encode(chunks, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    if faiss is None:
        st.warning("faiss is not available — similarity search will be slower (using dot product via numpy).")
        index = None
    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

    bm25 = BM25Okapi([c.split() for c in chunks])
    return index, bm25, embeddings


# =========================================================
# RETRIEVAL MODES
# =========================================================

def retrieve_standard(query, index, bm25, chunks, embeddings=None):
    # prefer faiss when available
    qv = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    if index is not None:
        _, idx = index.search(qv, top_k)
        return [chunks[i] for i in idx[0]]
    else:
        # fallback: compute cosine with numpy
        em_q = qv[0]
        sims = np.dot(embeddings, em_q)
        ids = np.argsort(-sims)[:top_k]
        return [chunks[i] for i in ids]


def retrieve_hyde(query, index, bm25, chunks, embeddings=None):
    if hf_client is None:
        st.warning("HyDE disabled: HF token / client not available. Falling back to Standard RAG.")
        return retrieve_standard(query, index, bm25, chunks, embeddings)

    hypothetic = hf_client.text_generation(
        model=MODEL_CHOICE,
        prompt=f"Write a short, factual answer to: {query}",
        max_new_tokens=80,
        temperature=0.2,
    )
    # API may return dict or list depending on client version
    hypothetic_text = hypothetic.get("generated_text") if isinstance(hypothetic, dict) else hypothetic[0].get("generated_text")

    qv = embed_model.encode([hypothetic_text], normalize_embeddings=True).astype("float32")
    if index is not None:
        _, idx = index.search(qv, top_k)
        return [chunks[i] for i in idx[0]]
    else:
        em_q = qv[0]
        sims = np.dot(embeddings, em_q)
        ids = np.argsort(-sims)[:top_k]
        return [chunks[i] for i in ids]


def generate_subqueries(q):
    if hf_client is None:
        # simple heuristic fallback
        terms = [t for t in re.split(r"[,;]\s*|\s-\s|\s\|\s|\s", q) if t]
        queries = [q] + terms[:3]
        return queries

    prompt = f"Generate 3 concise search queries (one per line) for: {q}\n"
    txt = hf_client.text_generation(model=MODEL_CHOICE, prompt=prompt, max_new_tokens=80, temperature=0.2)
    txt = txt.get("generated_text") if isinstance(txt, dict) else txt[0].get("generated_text")
    return [line.strip() for line in txt.splitlines() if line.strip()][:4]


def rrf(scores, k=60):
    final = {}
    for q, docs in scores.items():
        for rank, idx in enumerate(docs):
            final[idx] = final.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(final.items(), key=lambda x: x[1], reverse=True)


def retrieve_ragfusion(query, index, bm25, chunks, embeddings=None):
    sub_q = generate_subqueries(query)
    score_dict = {}

    for sq in sub_q:
        qv = embed_model.encode([sq], normalize_embeddings=True).astype("float32")
        if index is not None:
            _, idx = index.search(qv, top_k)
            score_dict[sq] = idx[0]
        else:
            em_q = qv[0]
            sims = np.dot(embeddings, em_q)
            ids = np.argsort(-sims)[:top_k]
            score_dict[sq] = ids

    fused = rrf(score_dict)
    top_ids = [i for i, _ in fused[:top_k]]
    return [chunks[i] for i in top_ids]


# Simple self-critique flow using the HF client if available
def self_critique(initial_answer, context, query):
    if hf_client is None:
        return initial_answer

    critique_prompt = f"You are a critic AI. QUESTION: {query}\nCONTEXT: {context}\nANSWER: {initial_answer}\nList any problems in the answer or write OK."
    critique = hf_client.text_generation(model=MODEL_CHOICE, prompt=critique_prompt, max_new_tokens=120, temperature=0.0)
    critique_text = critique.get("generated_text") if isinstance(critique, dict) else critique[0].get("generated_text")

    if "OK" in critique_text.upper():
        return initial_answer

    improve_prompt = f"Improve the answer based on this critique:\nCRITIQUE: {critique_text}\nOriginal Answer: {initial_answer}\nWrite an improved final answer."
    improved = hf_client.text_generation(model=MODEL_CHOICE, prompt=improve_prompt, max_new_tokens=200, temperature=0.2)
    improved_text = improved.get("generated_text") if isinstance(improved, dict) else improved[0].get("generated_text")
    return improved_text


# =========================================================
# FINAL ANSWER GENERATION
# =========================================================

def generate_answer(context, query):
    if hf_client is None:
        # lightweight fallback: echo context and say not found
        if query.lower() in context.lower():
            return context[:2000]
        return "The context does not provide this information. (HF token not configured)"

    prompt = f"Use ONLY the context below. If answer not found, say: \"The context does not provide this information.\"\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nFinal Answer:\n"
    resp = hf_client.text_generation(model=MODEL_CHOICE, prompt=prompt, max_new_tokens=250, temperature=0.2)
    resp_text = resp.get("generated_text") if isinstance(resp, dict) else resp[0].get("generated_text")
    return resp_text.split("Final Answer:")[-1].strip()


# =========================================================
# MAIN UI
# =========================================================
uploaded_pdf = st.file_uploader("📄 Upload PDF", type=["pdf"]) 
query = st.text_input("Ask a question:")

if uploaded_pdf is not None:
    with st.spinner("Processing PDF..."):
        text = extract_text(uploaded_pdf)
        chunks = chunk_text(text)
        index, bm25, embeddings = build_index(chunks)

    if not chunks:
        st.error("No text found in the uploaded PDF.")
    else:
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
                    st.write("- " + (c[:1000] + '...' if len(c) > 1000 else c))

else:
    st.info("Upload a PDF to begin. Tip: keep PDFs text-based (not scanned images) for best results.")

# ====== Footer ======
st.caption("SmartRAG — uses SentenceTransformers + optional HF Inference. Add HF token under Streamlit Secrets (HF_TOKEN).")

