import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

STORE_PATH = "pythontext_store.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_DIR = "./models/all-MiniLM-L6-v2"

# -------------------
# Load embeddings store (tuple only)
# -------------------
with open(STORE_PATH, "rb") as f:
    store = pickle.load(f)

# Expect tuple: (chunks, embeddings) or (chunks, embeddings, index)
if isinstance(store, tuple):
    chunks = store[0]
    embeddings = np.array(store[1])
    index = store[2] if len(store) > 2 else None
else:
    st.error("Store format must be a tuple (chunks, embeddings[, index])")
    st.stop()

# -------------------
# Load embedding model safely
# -------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set your HF token as environment variable
if os.path.exists(LOCAL_MODEL_DIR):
    model_path = LOCAL_MODEL_DIR
else:
    model_path = MODEL_NAME

embedder = SentenceTransformer(model_path, use_auth_token=HF_TOKEN)

# -------------------
# Load LLM pipeline
# -------------------
try:
    qa_pipeline = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3b-instruct",
        device_map="auto"
    )
    model_name = "meta-llama/Llama-3.2-3b-instruct"
except Exception as e:
    st.warning(f"Could not load LLaMA (reason: {e}). Falling back to smaller model.")
    qa_pipeline = pipeline(
        "text-generation",
        model="distilgpt2",
        device=-1
    )
    model_name = "distilgpt2"

# -------------------
# Retrieve top-k chunks
# -------------------
def retrieve(query, top_k=3):
    query_vec = embedder.encode([query])[0]
    scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_idx]

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Python Chatbot", layout="wide")
st.title("Python Textbook Chatbot")
st.write(f"Using **{model_name}** as backend")
st.write("Ask me questions from **pythontext.pdf**")

query = st.text_input("Your question:")

if query:
    # Retrieve context
    context_chunks = retrieve(query)
    context = "\n".join(context_chunks)

    # Create prompt
    prompt = f"""
Answer the following question using ONLY the context below.
If the context does not contain the answer, reply exactly:
'I could not find this in the text.'
Context:
{context}
Question: {query}
Answer:
"""

    # Generate response
    with st.spinner("Thinking..."):
        raw_output = qa_pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )[0]["generated_text"]

    # Extract answer only (after "Answer:" label)
    answer = raw_output.split("Answer:", 1)[-1].strip()

    st.write("### Answer:")
    st.write(answer)
