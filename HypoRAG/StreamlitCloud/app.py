# app.py
import streamlit as st
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="HyDE RAG", page_icon="🧠", layout="centered")

# -----------------------------
# Paths & Model Settings
# -----------------------------
BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"
FAISS_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "index_meta.pkl"

EMBED_MODEL = "sentence-transformers/sentence-t5-large"
RETRIEVE_TOP_K = 3
SCORE_THRESHOLD = 0.25
HYPOTHESIS_TOKENS = 128
ANSWER_TOKENS = 256

# -----------------------------
# Cached Model Loaders
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model():
    """Load Sentence Transformer embedding model."""
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def load_index():
    """Load FAISS index and metadata from local storage."""
    if not (FAISS_PATH.exists() and META_PATH.exists()):
        raise FileNotFoundError("FAISS index or metadata missing. Run index builder first.")
    index = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

@st.cache_resource(show_spinner=False)
def connect_hf_client():
    """Connect to Hugging Face model via API token."""
    token = st.secrets.get("HF_TOKEN", st.session_state.get("HF_TOKEN"))
    if not token:
        st.error("Missing HF Token. Please add it in Streamlit secrets.")
        st.stop()
    return InferenceClient(model="google/gemma-2-9b", token=token)

# -----------------------------
# Embedding and Search Functions
# -----------------------------
def get_embedding(text, model):
    """Generate normalized embedding vector for input text."""
    vector = model.encode([text], convert_to_numpy=True)
    vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)
    return vector.astype("float32")

def perform_search(index, meta, query_vec, top_k=RETRIEVE_TOP_K):
    """Search FAISS index for top-k most relevant results."""
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append((meta[idx]["text"], meta[idx]["source"], float(score)))
    return results

# -----------------------------
# Prompt Creation & Generation
# -----------------------------
def create_prompt(contexts, question):
    """Prepare the prompt for the final grounded answer."""
    joined_context = "\n\n---\n\n".join(contexts)
    return f"""
You are an intelligent assistant. Provide the answer using ONLY the following context.
If the answer is not available, respond exactly with: "The context does not provide this information."

CONTEXT:
{joined_context}

QUESTION:
{question}

ANSWER:
"""

def generate_hypothesis(client, question):
    """Generate a hypothetical answer (HyDE) to guide embedding-based retrieval."""
    hypo_prompt = f"""
Generate a short, hypothetical explanation (1–3 sentences) for the question below.
This will be used purely for retrieval and does not need to be accurate.

QUESTION: {question}

HYPOTHETICAL RESPONSE:
"""
    try:
        response = client.text_generation(
            hypo_prompt, max_new_tokens=HYPOTHESIS_TOKENS, do_sample=True, temperature=0.8
        )
    except Exception as e:
        st.error(f"Error while generating hypothetical text: {e}")
        return question

    if isinstance(response, dict):
        text = response.get("generated_text") or response.get("text") or str(response)
    elif isinstance(response, list) and len(response) > 0:
        first = response[0]
        if isinstance(first, dict):
            text = first.get("generated_text") or first.get("text") or str(first)
        else:
            text = str(first)
    else:
        text = str(response)

    return text.strip()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 HyDE RAG - Hypothetical Document Embedding Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.text_input("Ask a question about your document:")

if st.button("Submit") and user_query.strip():
    with st.spinner("Loading models and index..."):
        embedder = load_embed_model()
        index, meta = load_index()
        hf_client = connect_hf_client()

    # Step 1: Generate HyDE Document
    with st.spinner("Generating hypothetical document..."):
        hypothetical_doc = generate_hypothesis(hf_client, user_query)

    st.markdown("### 🧩 Generated Hypothetical Document")
    st.write(hypothetical_doc)

    # Step 2: Get Embedding for HyDE Text
    query_vec = get_embedding(hypothetical_doc, embedder)

    # Step 3: Retrieve Relevant Chunks
    retrieved = perform_search(index, meta, query_vec, top_k=RETRIEVE_TOP_K)
    filtered = [r for r in retrieved if r[2] >= SCORE_THRESHOLD]
    if not filtered and retrieved:
        filtered = retrieved
        st.info("Fallback: Using top results since no matches met the threshold.")

    st.markdown("### 📚 Retrieved Contexts")
    used_contexts = []
    if not filtered:
        st.warning("No relevant context found in the document.")
    else:
        for i, (text, src, score) in enumerate(filtered, start=1):
            st.write(f"{i}. {src} — Score: {score:.3f}")
            st.caption(text[:600] + ("..." if len(text) > 600 else ""))
            used_contexts.append((src, score))

    # Step 4: Generate Final Answer
    if not filtered:
        final_answer = "The context does not provide this information."
    else:
        context_list = [text for text, _, _ in filtered]
        prompt = create_prompt(context_list, user_query)

        with st.spinner("Generating grounded answer..."):
            try:
                response = hf_client.text_generation(
                    prompt, max_new_tokens=ANSWER_TOKENS, do_sample=False, temperature=0.2
                )
            except Exception as e:
                st.error(f"Error while generating answer: {e}")
                response = "The context does not provide this information."

            if isinstance(response, dict):
                final_answer = response.get("generated_text") or response.get("text") or str(response)
            elif isinstance(response, list) and len(response) > 0:
                first = response[0]
                final_answer = (
                    first.get("generated_text") if isinstance(first, dict) else str(first)
                )
            else:
                final_answer = str(response)

        final_answer = final_answer.strip() or "The context does not provide this information."

    # Step 5: Display Results
    st.markdown("### ✅ Final Answer")
    st.markdown(f"**Q:** {user_query}")
    st.markdown(f"**A:** {final_answer}")

    # Save Interaction
    st.session_state.history.append(
        {"question": user_query, "hypothesis": hypothetical_doc, "answer": final_answer, "contexts": used_contexts}
    )

# -----------------------------
# Show Conversation History
# -----------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 🕘 Conversation History")
    for i, entry in enumerate(st.session_state.history[:-1], 1):
        st.markdown(f"**Q{i}:** {entry['question']}")
        st.markdown(f"**A{i}:** {entry['answer']}")
        with st.expander("View Generated Hypothetical Document"):
            st.write(entry["hypothesis"])
        if entry.get("contexts"):
            with st.expander("View Retrieved Contexts"):
                for src, sc in entry["contexts"]:
                    st.write(f"- {src} — Score: {sc:.3f}")
        st.markdown("---")
