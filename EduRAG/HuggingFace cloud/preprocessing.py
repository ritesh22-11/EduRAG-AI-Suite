# preprocess_pdf.py
import pickle
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

PDF_PATH = "pythontext.pdf"
STORE_PATH = "pythontext_store.pkl"

def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def build_vector_store(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = NearestNeighbors(n_neighbors=5, metric="cosine").fit(embeddings)
    return {"chunks": chunks, "embeddings": embeddings, "index": index}

if __name__ == "__main__":
    print("Loading PDF...")
    text = load_pdf(PDF_PATH)

    print("Splitting into chunks...")
    chunks = chunk_text(text)

    print("Creating embeddings...")
    store = build_vector_store(chunks)

    print("Saving store...")
    with open(STORE_PATH, "wb") as f:
        pickle.dump(store, f)

    print("Done! Saved as pythontext_store.pkl")
