import streamlit as st
from pathlib import Path
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Fusion RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Fusion RAG Chatbot")

# =========================================================
# CONSTANTS
# =========================================================
HF_TOKEN = st.secrets["HF_TOKEN"]
APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"

EMBED_MODEL = "sentence-transformers/sentence-t5-large"
LLM_MODEL = "google/gemma-2-9b"
INDEX_NAME = "index"

# =========================================================
# CACHED RESOURCES — SAFE
# =========================================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def get_hf_client():
    return InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

# =========================================================
# FAISS LOADING 
# =========================================================
def load_faiss_or_fallback():
    emb = load_embeddings()

    faiss_path = FAISS_DIR / f"{INDEX_NAME}.faiss"
    pkl_path = FAISS_DIR / f"{INDEX_NAME}.pkl"

    if faiss_path.exists() and pkl_path.exists():
        # Load FAISS index
        db = FAISS.load_local(
            folder_path=str(FAISS_DIR),
            embeddings=emb,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True
        )

        # Extract text from FAISS internal docstore
        texts = []
        for doc_id in db.index_to_docstore_id.values():
            doc = db.docstore.search(doc_id)
            texts.append(doc.page_content)

        bm25 = BM25Okapi([t.split() for t in texts])
        return db, bm25, texts

    else:
        st.warning("⚠️ No FAISS index found. Using fallback documents.")

        fallback_texts = [
            "Fusion RAG retrieves documents with multiple queries.",
            "Reciprocal Rank Fusion merges results for better accuracy.",
            "This is a fallback document when no FAISS index is found."
        ]

        db = FAISS.from_texts(fallback_texts, emb)
        db.save_local(str(FAISS_DIR), INDEX_NAME)

        bm25 = BM25Okapi([t.split() for t in fallback_texts])
        return db, bm25, fallback_texts

@st.cache_resource
def load_faiss_index():
    return load_faiss_or_fallback()

# =========================================================
# LOAD GLOBALS
# =========================================================
db, bm25, stored_texts = load_faiss_index()
retriever = db.as_retriever(search_kwargs={"k": 4})
client = get_hf_client()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# FUSION RAG FUNCTIONS
# =========================================================
def generate_queries(q):
    return [
        q,
        f"Explain: {q}",
        f"What are the benefits of {q}?",
        f"What are challenges of {q}?",
        f"Give real-world applications of {q}"
    ]

def reciprocal_rank_fusion(results, k=60):
    scores = {}
    for q, docs in results.items():
        for rank, doc in enumerate(docs):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def fusion_rag_answer(query):
    exp_queries = generate_queries(query)
    results = {q: retriever.get_relevant_documents(q) for q in exp_queries}

    fused = reciprocal_rank_fusion(results)
    top_passages = [p for p, _ in fused[:5]]
    context = "\n\n".join(top_passages)

    prompt = f"""
Use the context ONLY. If answer is not in context, say:
"The context does not provide this information."

Context:
{context}

Question: {query}

Final Answer:
"""

    response = client.text_generation(
        prompt,
        max_new_tokens=350,
        temperature=0.2,
        do_sample=False
    )

    text = response.get("generated_text", "") if isinstance(response, dict) else str(response)
    if "Final Answer:" in text:
        return text.split("Final Answer:", 1)[-1].strip()
    return text.strip()

# =========================================================
# UI
# =========================================================
query = st.text_input("Ask me something:")

if query:
    with st.spinner("Thinking..."):
        answer = fusion_rag_answer(query)
        st.session_state.chat_history.append({"question": query, "answer": answer})

# =========================================================
# CHAT HISTORY
# =========================================================
if st.session_state.chat_history:
    st.subheader("📝 Conversation History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")
        st.markdown("---")
