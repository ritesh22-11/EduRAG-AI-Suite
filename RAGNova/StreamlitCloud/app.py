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

# ---------- Basic config ----------
st.set_page_config(page_title="RAGNova Fusion", page_icon="🤖", layout="centered")
st.title("🤖 RAGNova — Fusion Retrieval + Generation ")

APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
DEFAULT_EMBED = "sentence-transformers/sentence-t5-large"
DEFAULT_LLM = "microsoft/Phi-3-mini-4k-instruct"

# ---------- UI: Model & Advanced Settings ----------
with st.sidebar:
    st.header("Model & Settings")
    model_choice = st.selectbox(
        "Choose LLM (inference)",
        [
            "microsoft/Phi-3-mini-4k-instruct",
            "tiiuae/Falcon3-3B-Instruct",
            "meta-llama/Llama-3.2-3b-instruct",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ],
        index=0
    )
    embed_model = st.selectbox("Embedding model", [DEFAULT_EMBED, "sentence-transformers/all-MiniLM-L12-v2"], index=0)

    top_k = st.slider("Top chunks to pass to model", 1, 10, 5)
    bm25_weight = st.slider("BM25 weight", 0.0, 1.0, 0.6)
    dense_weight = st.slider("Dense (FAISS) weight", 0.0, 1.0, 0.4)
    max_context_chars = st.number_input("Max combined context characters", min_value=1000, max_value=60000, value=6000, step=500)
    max_gen_tokens = st.slider("Max new tokens (generation)", 50, 600, 300)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    st.markdown("---")
    st.markdown("Tip: Use Phi / Falcon on CPU. Llama requires HF token and may be slow without GPU.")
    if not HF_TOKEN:
        st.info("HF token not found in secrets — gated models will require it.")

# ---------- Caches / Resource loaders ----------
@st.cache_resource(show_spinner=True)
def get_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})

@st.cache_resource(show_spinner=True)
def get_hf_client(model_name: str):
    # Use token if present; InferenceClient handles None token as public
    return InferenceClient(model=model_name, token=HF_TOKEN or None)

def _local_faiss_paths() -> Tuple[Path, Path]:
    return FAISS_DIR / "index.faiss", FAISS_DIR / "index_meta.pkl"

def has_local_faiss() -> bool:
    idx, meta = _local_faiss_paths()
    return idx.exists() and meta.exists()

def build_bm25_from_texts(texts: List[str]) -> BM25Okapi:
    tokenized = [t.split() for t in texts]
    return BM25Okapi(tokenized)

@st.cache_resource(show_spinner=True)
def load_or_create_faiss(embeddings):
    """
    If FAISS exists, load it. Otherwise create demo fallback index.
    """
    if has_local_faiss():
        # Use LangChain loader (assumes files in faiss_index/)
        db = FAISS.load_local(folder_path=str(FAISS_DIR), embeddings=embeddings, index_name=INDEX_NAME, allow_dangerous_deserialization=True)
        # extract texts for BM25
        # LangChain FAISS stores texts in db.docstore; fallback if missing
        try:
            docs = [db.docstore[i] for i in range(len(db.docstore))]
            texts = [d if isinstance(d, str) else d.page_content for d in docs]
        except Exception:
            # fallback: load metadata if present
            meta_path = FAISS_DIR / "index_meta.pkl"
            texts = []
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                    texts = [m.get("text", "") for m in meta]
        bm25 = build_bm25_from_texts(texts)
        return db, bm25, texts
    else:
        # fallback demo index (user should upload/build)
        demo_texts = [
            "Fusion RAG merges results from multiple queries for robust retrieval.",
            "Reciprocal Rank Fusion (RRF) combines rankings from different retrieval sources.",
            "This demo index is temporary. Upload a PDF and click 'Build index' to create your own."
        ]
        db = FAISS.from_texts(demo_texts, embeddings)
        db.save_local(str(FAISS_DIR), index_name=INDEX_NAME)
        bm25 = build_bm25_from_texts(demo_texts)
        return db, bm25, demo_texts

emb = get_embeddings(embed_model)
db, bm25, stored_texts = load_or_create_faiss(emb)
retriever = db.as_retriever(search_kwargs={"k": top_k * 2})  # get a little more to fuse later
client = get_hf_client(model_choice)

# ---------- Helper functions ----------
def generate_expanded_queries(q: str) -> List[str]:
    return [
        q,
        f"Explain in detail: {q}",
        f"What are the benefits of {q}?",
        f"What are the challenges or drawbacks of {q}?",
        f"Give a real-world application of {q}"
    ]

def reciprocal_rank_fusion(results: Dict[str, List], k: int = 60, bw=0.6, dw=0.4) -> List[Tuple[str, float]]:
    """
    results: dict(query -> list of docs) where each doc has .page_content or is a raw string
    returns: list of (text, score) sorted desc
    """
    scores = {}
    for q, docs in results.items():
        for rank, d in enumerate(docs):
            text = d.page_content if hasattr(d, "page_content") else str(d)
            scores[text] = scores.get(text, 0.0) + (dw / (k + rank + 1))
    # BM25 contribution (rank by bm25 scores already computed externally if available)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def trim_context(context: str, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    # keep full sentences until max_chars
    sentences = re.split(r'(?<=[.!?])\s+', context)
    out = ""
    for s in sentences:
        if len(out) + len(s) + 2 > max_chars:
            break
        out += s + " "
    return out.strip()

def build_prompt(context: str, question: str) -> str:
    return (
        "You are a friendly, accurate study assistant. "
        "Answer ONLY using the provided CONTEXT. "
        "If the information is not present in the context, reply exactly: "
        "\"The context does not provide this information.\"\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely in 3-7 sentences:"
    )

def extract_generated_text(response) -> str:
    # InferenceClient returns dict or list depending on model; standardize
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("generated_text", next(iter(response.values()), ""))
    if isinstance(response, list):
        if len(response) > 0:
            return response[0].get("generated_text", str(response[0]))
    return str(response)

# ---------- UI: upload / build index ----------
st.markdown("### Data / Index")
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_pdf = st.file_uploader("Upload book PDF to build FAISS index (optional)", type=["pdf"])
with col2:
    if st.button("Build index from uploaded PDF"):
        if not uploaded_pdf:
            st.warning("Upload a PDF first.")
        else:
            # Build index: extract, chunk, embed, write index + metadata
            with st.spinner("Building FAISS index (this can take a few minutes)..."):
                from PyPDF2 import PdfReader
                def extract_text_from_pdf_file(file_stream) -> str:
                    reader = PdfReader(file_stream)
                    pages = [p.extract_text() or "" for p in reader.pages]
                    return "\n\n".join(pages)
                text = extract_text_from_pdf_file(uploaded_pdf)
                # chunk by ~800 chars with 150 overlap
                CHUNK_SIZE = 800
                OVERLAP = 150
                chunks = []
                i = 0
                N = len(text)
                while i < N:
                    j = min(i + CHUNK_SIZE, N)
                    chunks.append(text[i:j].strip())
                    i = j - OVERLAP if j < N else j
                # embed in batches
                batch_size = 32
                all_emb = []
                sentencemodel = emb
                for start in range(0, len(chunks), batch_size):
                    batch = chunks[start:start+batch_size]
                    batch_emb = sentencemodel.embed_documents(batch)  # langchain embedding API
                    all_emb.extend(batch_emb)
                arr = np.asarray(all_emb, dtype="float32")
                # normalize
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                arr = arr / norms
                dim = arr.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(arr)
                # save
                FAISS_DIR.mkdir(exist_ok=True)
                faiss.write_index(index, str(FAISS_DIR / "index.faiss"))
                meta = [{"text": chunks[i], "source": f"uploaded#{i}"} for i in range(len(chunks))]
                with open(FAISS_DIR / "index_meta.pkl", "wb") as f:
                    pickle.dump(meta, f)
                # also save via LangChain wrapper for easier loading
                db2 = FAISS.from_texts(chunks, emb)
                db2.save_local(str(FAISS_DIR), index_name=INDEX_NAME)
                st.success("Index built and saved to faiss_index/ successfully. Reload page to load it.")
                st.experimental_rerun()

# ---------- UI: Query & Chat ----------
st.markdown("---")
query = st.text_input("Ask anything about the indexed book or content:")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if st.button("Clear chat history"):
    st.session_state["chat_history"] = []

if query:
    with st.spinner("Running Fusion-RAG retrieval..."):
        # 1. generate expanded queries
        expanded = generate_expanded_queries(query)
        # 2. for each q, retrieve candidate docs (FAISS) and compute BM25 ranks
        results = {}
        for q in expanded:
            docs = retriever.get_relevant_documents(q)
            results[q] = docs
        # 3. compute BM25 scores on stored_texts if available
        bm25_scores = {}
        if stored_texts:
            bm25_obj = build_bm25_from_texts(stored_texts)
            # compute list of (index, score) sorted descending
            bm25_scores = bm25_obj.get_scores(query.split())
        # 4. fuse using reciprocal_rank_fusion (we created a simple RRF via results)
        fused = reciprocal_rank_fusion(results, k=60, bw=bm25_weight, dw=dense_weight)
        # fused is list of (text, score)
        top_texts = [t for t, _ in fused][: top_k * 2]
        # deduplicate and keep top_k after trimming
        seen = set()
        final_chunks = []
        for txt in top_texts:
            if txt in seen:
                continue
            seen.add(txt)
            final_chunks.append(txt)
            if len(final_chunks) >= top_k:
                break
        context = "\n\n".join(final_chunks)
        context = trim_context(context, max_context_chars)

    # 5. Generate prompt and call LLM
    prompt = build_prompt(context, query)
    with st.spinner(f"Calling model {model_choice.split('/')[-1]} ..."):
        try:
            response = client.text_generation(
                prompt,
                max_new_tokens=max_gen_tokens,
                temperature=temperature,
                do_sample=False,
                stream=False,
            )
            plain = extract_generated_text(response)
        except Exception as e:
            st.error(f"Model call failed: {e}")
            plain = "The model call failed. Check logs."

    st.session_state["chat_history"].append({"q": query, "a": plain, "ctx": final_chunks})

# ---------- Display history and context ----------
if st.session_state["chat_history"]:
    st.markdown("## Conversation History")
    for i, item in enumerate(reversed(st.session_state["chat_history"]), 1):
        st.markdown(f"**Q{i}:** {item['q']}")
        st.markdown(f"**A{i}:** {item['a']}")
        with st.expander("Show retrieved context used for this answer", expanded=False):
            for j, c in enumerate(item.get("ctx", []), 1):
                st.markdown(f"**Chunk {j}:** {c[:1000]}{'...' if len(c) > 1000 else ''}")
        st.markdown("---")

# ---------- Optional: Download index / metadata snapshot ----------
st.markdown("---")
st.markdown("Index management:")
if st.button("Download faiss_index zip (for backup)"):
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in FAISS_DIR.glob("*"):
            z.write(p, arcname=p.name)
    buf.seek(0)
    st.download_button("Download faiss_index.zip", data=buf, file_name="faiss_index.zip")

st.caption("Built RAGNova ")
