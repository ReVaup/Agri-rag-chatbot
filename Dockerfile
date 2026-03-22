# =========================================
# Agricultural RAG Chatbot — Dockerfile
# Python 3.11 slim, CPU inference
# Ollama must run separately on the host
# =========================================
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_8.py .
COPY chunking_4.py .
COPY embedding_5.py .
COPY qdrant_6.py .
COPY eval.py .

# Copy pre-built knowledge base
COPY chunks.pkl .
COPY embeddings.pkl .

# Gradio runs on 7860
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app_8.py"]
