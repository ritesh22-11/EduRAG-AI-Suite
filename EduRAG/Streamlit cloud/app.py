import streamlit as st
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from huggingface_hub import InferenceClient
import os

# -----------------------------
# PDF Text Loader & Chunker
# -----------------------------
def load_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def chunk_text(text, max_tokens=200):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk, current_len = [], [], 0
    for s in sentences:
        wc = len(s.split())
        if current_len + wc > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [s], wc
        else:
            current_chunk.append(s); current_len += wc
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# -----------------------------
# Vector Store with FAISS
# -----------------------------
class SimpleVectorStore:
    def __init__(self, dim):
        self.dim, self.vectors, self.metadata = dim, [], []
        self.index = None

    def add(self, vs, metas):
        for v, m in zip(vs, metas):
            self.vectors.append(np.array(v, dtype=np.float32))
            self.metadata.append(m)
        if self.vectors:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(np.stack(self.vectors))

    def search(self, qv, k=5):
        qv = np.array(qv, dtype=np.float32).reshape(1, -1)
        _, I = self.index.search(qv, k)
        return [self.metadata[i] for i in I[0]]

# -----------------------------
# Build Embeddings & Index
# -----------------------------
def index_pdf(uploaded_file):
    text = load_pdf_text(uploaded_file)
    chunks = chunk_text(text)
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embed.encode(chunks)
    store = SimpleVectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    return embed, store

# -----------------------------
# Hugging Face Chat Client
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
LLM_MODEL = "google/gemma-2-2b-it"
client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

def query_hf_api(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful tutor."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat_completion(messages=messages, max_tokens=300)
    return response["choices"][0]["message"]["content"]

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Student Assisted Chatbot", page_icon="🤖", layout="wide")
st.title("🎓 Student Assisted Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
user_input = st.text_input("Your question:")

if uploaded_file and user_input:
    try:
        embed_model, store = index_pdf(uploaded_file)
        query_vec = embed_model.encode([user_input])[0]
        relevant_chunks = store.search(query_vec, k=5)
        context = "\n".join(relevant_chunks)

        prompt = f"""
Based only on the following context, answer the question in full sentences.
If the context does not contain enough information, say "I could not find this in the text."

Context:
{context}

Question: {user_input}
Answer:
"""

        answer = query_hf_api(prompt)

        st.subheader("🧠 Answer")
        st.write(answer)
    except Exception as e:
        st.error(f"Error: {e}")
