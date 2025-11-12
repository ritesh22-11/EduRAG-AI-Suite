FROM python:3.10-slim

LABEL maintainer="Ritesh Kumar <riteshkumar48171@gmail.com>"
LABEL project="RAGNova - Intelligent Document Retrieval and Q&A System"
LABEL version="1.0"
LABEL license="MIT"
LABEL description="A professional Retrieval-Augmented Generation (RAG) app built with Streamlit, FAISS, and Llama 3."

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV PATH="/home/raguser/.local/bin:$PATH"

RUN useradd -m -u 1000 raguser
USER raguser

WORKDIR /app

COPY --chown=raguser ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=raguser . /app

ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=300
ENV STREAMLIT_RUNTIME_TMP_DIR=/app/tmp

RUN mkdir -p /app/tmp && chmod 777 /app/tmp

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
