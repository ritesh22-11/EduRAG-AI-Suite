# build_index.py  
import pickle
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# =============================
# CONFIG
# =============================
HERE = Path(__file__).parent
PDF_PATH = HERE / "d2l-en.pdf"        # Place your PDF here
OUT_DIR = HERE / "faiss_index"
MODEL_NAME = "sentence-transformers/sentence-t5-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except:
            pages.append("")
    return "\n\n".join(pages)


def chunk_text(text: str) -> List[str]:
    clean = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(clean)

    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP if end < n else n

    return chunks


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"{PDF_PATH} not found!")

    print("Extracting text...")
    text = extract_text(PDF_PATH)

    print("Chunking...")
    chunks = chunk_text(text)
    print("Chunks:", len(chunks))

    print("Loading embeddings...")
    embedder = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    print("Building FAISS index through LangChain...")
    db = FAISS.from_texts(chunks, embedder)

    OUT_DIR.mkdir(exist_ok=True)

    print("Saving FAISS index...")
    db.save_local(str(OUT_DIR), index_name="index")

    print("\n========================================")
    print("SUCCESS! Generated FAISS index:")
    print(f"- {OUT_DIR/'index.faiss'}")
    print(f"- {OUT_DIR/'index.pkl'}")
    print("========================================")


if __name__ == "__main__":
    main()
