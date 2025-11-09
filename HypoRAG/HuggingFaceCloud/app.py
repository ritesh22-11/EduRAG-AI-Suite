import os
import gradio as gr
import fitz  # PyMuPDF for PDFs
import docx
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from gtts import gTTS
from huggingface_hub import login

# 1) Authentication & Model Config

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Please set your HF_TOKEN as an environment variable before running the app.")

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = "meta-llama/Llama-3.2-3b-instruct"
ASR_MODEL_ID = "openai/whisper-small"

# 2) Load Models
embedding_model = SentenceTransformer(EMBED_MODEL_ID)

login(HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=HF_TOKEN)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN
)

# Whisper ASR Model
stt_model = pipeline("automatic-speech-recognition", model=ASR_MODEL_ID, token=HF_TOKEN)

# 3) Read File Content
def read_file_text(file_path: str) -> str:
    """Extract readable text from PDF, DOCX, or TXT file."""
    if not file_path:
        return ""
    ext = os.path.splitext(file_path.lower())[1]
    text = ""
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text")
    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text

# 4) Create Vector Index (FAISS)
def create_vector_index(text: str, chunk_len=500, overlap=50):
    """Split long text into chunks, embed them, and build FAISS index."""
    if not text.strip():
        return None, None

    pieces = []
    step = max(1, chunk_len - overlap)
    for i in range(0, len(text), step):
        snippet = text[i:i + chunk_len].strip()
        if snippet:
            pieces.append(snippet)

    if not pieces:
        return None, None

    embeddings = embedding_model.encode(pieces, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, pieces

# 5) Global Index Variables
doc_index = None
doc_chunks = None

# 6) File Upload Handler
def handle_upload(file_path: str):
    """Read document, create FAISS index, and confirm chunk count."""
    global doc_index, doc_chunks
    if not file_path:
        return "Please upload a file first."
    text = read_file_text(file_path)
    index, chunks = create_vector_index(text)
    if index is None:
        return "Could not index file (empty or unsupported)."
    doc_index, doc_chunks = index, chunks
    return f"Document indexed successfully — {len(chunks)} chunks stored."

# 7) Query Handling
def process_query(question: str):
    """Retrieve relevant text chunks and generate an LLM-based answer."""
    global doc_index, doc_chunks
    if not question.strip():
        return "Please enter a valid question."
    if doc_index is None or not doc_chunks:
        return "Please upload and index a document first."

    # Step 1: Retrieve Relevant Chunks
    q_vec = embedding_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    _, I = doc_index.search(q_vec, k=min(5, len(doc_chunks)))
    retrieved = [doc_chunks[i] for i in I[0]]
    context = "\n".join(retrieved)

    # Step 2: Generate Answer using LLM
    prompt = f"""
    [INST] You are a helpful educational assistant. 
    Use only the context below to answer.
    If the answer is not found, say "I could not find this in the text."
    Context:
    {context}
    Question: {question}
    Answer: [/INST]
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(llm.device)
    outputs = llm.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1].strip()
    return decoded

# 8) Text-to-Speech
def synthesize_with_gtts(text: str, out_path="answer.mp3"):
    """Convert generated text answer to speech using gTTS."""
    tts = gTTS(text=text, lang="en")
    tts.save(out_path)
    return out_path

# 9) Voice Query Handler
def handle_voice_query(audio_path: str):
    """Convert speech to text, process query, and return spoken answer."""
    if not audio_path:
        return "Please record your question.", "", None

    asr_result = stt_model(audio_path)
    recognized_text = asr_result.get("text", "").strip()
    if not recognized_text:
        return "Could not transcribe audio.", "", None

    answer = process_query(recognized_text)
    audio_output = synthesize_with_gtts(answer, "response.mp3")
    return recognized_text, answer, audio_output

# 10) Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan")) as demo:
    gr.Markdown("# 🤖 EduRAG Voice-Enabled Chatbot")
    gr.Markdown("Upload a document and ask your questions by typing or speaking!")

    with gr.Row():
        # Left Panel: File Upload
        with gr.Column(scale=1):
            file_input = gr.File(label="📂 Upload File (PDF/DOCX/TXT)", type="filepath")
            index_btn = gr.Button("🧩 Index Document", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False)

        # Right Panel: Interaction
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Text Chat")
            question_box = gr.Textbox(label="Enter Question", placeholder="e.g., Summarize this topic")
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer_box = gr.Textbox(label="Answer", lines=7)

            gr.Markdown("### 🎤 Voice Chat")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak your question")
            transcribed_box = gr.Textbox(label="Recognized Speech", interactive=False)
            voice_answer_box = gr.Textbox(label="Voice Answer", lines=7)
            voice_output_audio = gr.Audio(label="Bot Voice Output")

    # Connect Buttons
    index_btn.click(fn=handle_upload, inputs=file_input, outputs=status_box)
    ask_btn.click(fn=process_query, inputs=question_box, outputs=answer_box)
    audio_input.change(fn=handle_voice_query,
                       inputs=audio_input,
                       outputs=[transcribed_box, voice_answer_box, voice_output_audio])

# 11) Launch Application
demo.launch()
