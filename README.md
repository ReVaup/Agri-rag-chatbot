# Agricultural RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot built on 63 agricultural research papers from the ICRISAT Open Access Repository. Ask questions about crop genetics, drought resistance, domestication, and more — and get answers grounded in peer-reviewed research.

---

## Demo

> **"What genes are responsible for pod shattering resistance in soybean?"**
>
> The chatbot retrieves the relevant chunk from *Frontiers in Genetics (2022)* and answers:
> *qPHD1 promotes pod dehiscence by modulating cell-wall lignin in inner sclerenchyma. SHAT1-5, a NAC gene similar to A. thaliana NST1, confers pod-breaking resistance by stimulating lignification of fiber cap cells in pod sutures.*

---

## Architecture

```
PDFs (63 papers)
    ↓  scrapper_1.py + pdf_download_2.py
Raw PDFs
    ↓  pdf_to_markdown_3.py  (Docling)
Markdown files
    ↓  chunking_4.py
Text chunks + Table chunks  (7,991 total)
    ↓  embedding_5.py  (all-MiniLM-L6-v2, CUDA)
Embeddings (7991 × 384, float32, normalized)
    ↓  qdrant_6.py
Qdrant vector store
    ↓  app_8.py  (hybrid search + LLaMA 3.2 3B via Ollama)
Gradio chatbot UI
```

---

## Features

- **ETL pipeline** — scrapes ICRISAT OAR, downloads 63 PDFs, converts to markdown via Docling
- **Smart chunking** — separates text and tables, preserves table titles, cross-section overlap bridging, noise filtering (affiliations, references, image placeholders)
- **GPU embeddings** — `all-MiniLM-L6-v2` on CUDA, 384-dim normalized vectors
- **Hybrid search** — semantic similarity + normalized keyword re-ranking + table chunk boost
- **Streaming responses** — word-by-word output via Ollama `stream=True`
- **Multi-turn memory** — full conversation history (user + assistant) passed to LLaMA
- **Logging** — per-query latency and token usage logged to `rag_chatbot.log`
- **Evaluation** — automated precision@5 + term recall + latency measurement across 20 queries

---

## Evaluation Results

| Metric | Result | Target |
|---|---|---|
| Precision@5 | **90.0%** | >80% |
| Avg term recall | **87.5%** | — |
| Avg retrieval latency | **0.040s** | <5s |
| P95 retrieval latency | **0.062s** | — |
| All 20 queries under 5s | **20/20** | — |

---

## Project Structure

```
agri-rag-chatbot/
├── scrapper_1.py          # Scrapes ICRISAT OAR for PDF links
├── pdf_download_2.py      # Downloads PDFs from scraped links
├── pdf_to_markdown_3.py   # Converts PDFs to markdown via Docling
├── chunking_4.py          # Smart chunking — text + table separation
├── embedding_5.py         # GPU embedding generation
├── qdrant_6.py            # Uploads vectors to Qdrant
├── app_8.py               # Gradio chatbot with hybrid search + LLaMA
├── eval.py                # Evaluation script — precision@5, recall, latency
├── Dockerfile             # Container for the chatbot
├── docker-compose.yml     # Runs chatbot + Qdrant together
├── requirements.txt       # Pinned Python dependencies
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- LLaMA 3.2 pulled: `ollama pull llama3.2`
- [Qdrant](https://qdrant.tech) running locally on port 6333
- NVIDIA GPU recommended (RTX 3060 or better)

### 1. Clone the repo

```bash
git clone https://github.com/ReVaup/agri-rag-chatbot.git
cd agri-rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# For GPU (CUDA 12.1):
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Run the ETL pipeline (skip if using pre-built data)

```bash
python scrapper_1.py        # scrape PDF links
python pdf_download_2.py    # download PDFs
python pdf_to_markdown_3.py # convert to markdown
python chunking_4.py        # chunk documents
python embedding_5.py       # generate embeddings
python qdrant_6.py          # upload to Qdrant
```

### 4. Launch the chatbot

```bash
python app_8.py
```

Open `http://localhost:7860` in your browser.

### 5. Run evaluation

```bash
python eval.py
```

---

## Docker

### Build and run with Docker Compose

```bash
docker-compose up --build
```

This starts Qdrant on port 6333 and the chatbot on port 7860.

> **Note:** Ollama must be running on the host machine before starting the container. The container connects to it via `host.docker.internal`.

---

## Sample Questions

- *What genes are responsible for pod shattering resistance in soybean?*
- *How does drought stress affect SiPRX gene expression in sesame?*
- *What are the key domestication syndrome traits in food legumes?*
- *What QTLs are associated with root-knot nematode resistance in rice?*
- *What is the role of nitrogen fixation in legume crops?*
- *What are the major subfamilies of the Leguminosae family?*

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.2 3B via Ollama |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector DB | Qdrant |
| PDF parsing | Docling |
| UI | Gradio |
| GPU | NVIDIA RTX 3060 (CUDA 12.1) |

---

## Dataset

63 peer-reviewed agricultural research papers sourced from the [ICRISAT Open Access Repository](https://oar.icrisat.org), covering topics including:
- Crop genetics and genomics
- Drought and heat stress tolerance
- Legume domestication and evolution
- Millet nutritional profiles
- Molecular markers and QTL mapping
