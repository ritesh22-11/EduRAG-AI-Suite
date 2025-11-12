import os
import re
import time
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

#  PDF Processing Utilities

def extract_text_from_pdf(uploaded_pdf) -> str:
    """
    Extract plain text from each page of an uploaded PDF.
    Returns a concatenated text string.
    """
    pdf_document = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text_content = ""
    for page in pdf_document:
        text_content += page.get_text()
    pdf_document.close()

    if not text_content.strip():
        raise ValueError("No readable text found in the PDF file.")
    return text_content.strip()


def split_text_into_chunks(text: str, max_words: int = 200):
    """
    Split a long text into smaller semantic chunks based on sentences.
    Each chunk is limited by a rough word count (max_words).
    """
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text)
    chunks, current_chunk, word_count = [], [], 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if word_count + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk, word_count = [sentence], len(words)
        else:
            current_chunk.append(sentence)
            word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Filter out overly small or blank chunks
    filtered_chunks = [chunk for chunk in chunks if len(chunk.split()) >= 5]
    if not filtered_chunks:
        raise ValueError("The PDF did not produce valid text segments.")
    return filtered_chunks


#  Retrieval System (FAISS + BM25)

class HybridRetriever:
    """
    Combines dense vector search (FAISS) and sparse keyword search (BM25)
    and fuses their results using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.encoder = SentenceTransformer(embedding_model)
        self.vector_index = None
        self.bm25_index = None
        self.text_segments = []
        self.vector_dim = None

    def build_index(self, text_segments):
        """Build FAISS and BM25 indices from provided text segments."""
        self.text_segments = list(text_segments)
        # BM25 setup
        tokenized_docs = [segment.split() for segment in self.text_segments]
        self.bm25_index = BM25Okapi(tokenized_docs)

        # FAISS setup
        embeddings = self.encoder.encode(self.text_segments, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        self.vector_dim = embeddings.shape[1]

        self.vector_index = faiss.IndexFlatIP(self.vector_dim)
        self.vector_index.add(embeddings)

    def retrieve(self, user_query: str, top_k: int = 5, fusion_k: int = 60):
        """
        Retrieve relevant passages using FAISS and BM25,
        then fuse results via Reciprocal Rank Fusion.
        """
        if self.vector_index is None or self.bm25_index is None:
            raise RuntimeError("Index not initialized. Please build it first.")

        # Dense retrieval
        query_vec = self.encoder.encode([user_query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        sim_scores, sim_indices = self.vector_index.search(query_vec, min(top_k * 5, len(self.text_segments)))
        dense_ranked = sim_indices[0].tolist()

        # Sparse retrieval
        bm25_scores = self.bm25_index.get_scores(user_query.split())
        sparse_ranked = np.argsort(-bm25_scores).tolist()

        # Reciprocal Rank Fusion (RRF)
        rank_scores = {}
        for rank, idx in enumerate(dense_ranked):
            rank_scores[idx] = rank_scores.get(idx, 0) + 1 / (fusion_k + rank + 1)
        for rank, idx in enumerate(sparse_ranked):
            rank_scores[idx] = rank_scores.get(idx, 0) + 1 / (fusion_k + rank + 1)

        combined = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in combined[:top_k]]
        return [self.text_segments[i] for i in top_indices]


#  LLM Setup (Llama 3.2 3B Instruct)

@st.cache_resource(show_spinner=True)
def load_model(model_name: str = "meta-llama/Llama-3.2-3b-instruct"):
    """
    Load a gated Hugging Face model using authentication token (HF_TOKEN).
    """
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("Hugging Face token not found. Add HF_TOKEN in your Hugging Face Space secrets.")

    login(token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def generate_response(tokenizer, model, retrieved_text: str, question: str, max_new_tokens: int = 300):
    """
    Generate a response using the LLM with context-based reasoning.
    """
    dialogue = [
        {
            "role": "system",
            "content": (
                "You are a concise academic assistant. "
                "Only answer based on the provided context. "
                "If context is insufficient, respond exactly: 'Information not found in the document.'"
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{retrieved_text}\n\nQuestion: {question}\nAnswer clearly in 3–7 sentences.",
        },
    ]

    prompt = tokenizer.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response.strip()


#  Streamlit Frontend

st.set_page_config(page_title="RAGNova - Smart Document Assistant", page_icon="🤖", layout="centered")
st.title("🤖 RAGNova— Intelligent Document Retrieval and Q&A")

with st.expander("How this works"):
    st.markdown(
        """
        1. Upload a textbook or document in PDF format.  
        2. The system extracts text and creates hybrid search indexes (BM25 + FAISS).  
        3. The **Reciprocal Rank Fusion** algorithm selects the most relevant chunks.  
        4. The **Llama-3.2-3B-Instruct** model generates an accurate, context-grounded answer.
        """
    )

uploaded_pdf = st.file_uploader("📄 Upload your PDF document", type=["pdf"])
user_query = st.text_input("💬 Ask a question about your document")

@st.cache_resource(show_spinner=True)
def build_retriever_from_pdf(pdf_bytes: bytes):
    """Convert PDF → text → chunks → hybrid retriever."""
    text_data = extract_text_from_pdf(uploaded_pdf=type("Tmp", (), {"read": lambda self=None: pdf_bytes})())
    chunks = split_text_into_chunks(text_data)
    retriever = HybridRetriever()
    retriever.build_index(chunks)
    return retriever, chunks


if uploaded_pdf and user_query:
    try:
        pdf_data = uploaded_pdf.read()
        retriever, all_chunks = build_retriever_from_pdf(pdf_data)

        with st.spinner("🔍 Searching relevant sections..."):
            relevant_chunks = retriever.retrieve(user_query, top_k=5)
        combined_context = "\n\n".join(f"- {chunk}" for chunk in relevant_chunks)

        tokenizer, model = load_model()

        with st.spinner("🤖 Generating answer..."):
            response = generate_response(tokenizer, model, combined_context, user_query)

        st.subheader("🧠 Generated Answer")
        st.write(response)

        with st.expander("📚 View Retrieved Context"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}**\n\n{chunk}")

    except Exception as err:
        st.error(f"⚠️ Error: {err}")

else:
    st.info("Please upload a PDF and type a question to begin.")
