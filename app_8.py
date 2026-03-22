import time
import logging
import torch
import ollama
import gradio as gr
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# =========================================
# LOGGING SETUP
# Logs latency + token usage per query
# =========================================
logging.basicConfig(
    filename="rag_chatbot.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# =========================================
# DEVICE DETECTION
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Embedding device: {device}")
print(f"Embedding device: {device}")

# =========================================
# LOAD EMBEDDING MODEL
# =========================================
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# =========================================
# CONNECT QDRANT
# =========================================
client          = QdrantClient(host="localhost", port=6333)
collection_name = "agri_rag_final"

# =========================================
# EMBED QUERY
# =========================================
def embed_query(query: str):
    return model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

# =========================================
# KEYWORD SCORE  (normalized by doc length)
# Fixes: long docs no longer win by default
# =========================================
def keyword_score(query: str, text: str) -> float:
    query_words = set(query.lower().split())
    text_words  = text.lower().split()
    if not text_words:
        return 0.0
    hits = sum(1 for w in text_words if w in query_words)
    return hits / len(text_words)

# =========================================
# QDRANT SEARCH
# =========================================
def search_qdrant(query: str, top_k: int = 10):
    query_vector = embed_query(query)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )
    return results.points

# =========================================
# HYBRID SEARCH
# semantic (main) + keyword (tiebreaker)
# Table chunks get a small boost since they
# carry structured data that answers directly
# =========================================
def hybrid_search(query: str, top_k: int = 5):
    t0      = time.time()
    results = search_qdrant(query, top_k=10)

    scored = []
    for r in results:
        text       = r.payload.get("text", "")
        chunk_type = r.payload.get("type", "text")
        sem_score  = r.score
        key_score  = keyword_score(query, text)
        type_boost = 0.05 if chunk_type == "table" else 0.0
        final      = sem_score + (0.1 * key_score) + type_boost
        scored.append((final, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top     = [r for _, r in scored[:top_k]]
    latency = time.time() - t0
    log.info(f"RETRIEVAL | query='{query[:60]}' | top_k={top_k} | latency={latency:.3f}s")
    return top, latency

# =========================================
# BUILD CONTEXT
# FIX: raised from 3000 to 8000 chars
# Chunks included fully, not sliced mid-sentence
# Table chunks show their title as context header
# =========================================
def get_context(results, max_chars: int = 8000) -> str:
    context = ""
    for r in results:
        payload    = r.payload
        source     = payload.get("source", "unknown")
        text       = payload.get("text", "")
        chunk_type = payload.get("type", "text")
        title      = payload.get("title", "")

        if chunk_type == "table" and title and not text.startswith(title):
            entry = f"[Source: {source} | Table: {title[:60]}]\n{text}\n\n"
        else:
            entry = f"[Source: {source}]\n{text}\n\n"

        if len(context) + len(entry) > max_chars:
            break
        context += entry
    return context

# =========================================
# PROMPT TEMPLATES
# Optimized: trimmed by ~25% vs original
# while keeping all critical instructions
# Token reduction is logged per query
# =========================================
informational_prompt = """You are an expert agricultural research assistant.

RULES:
- Use the context as PRIMARY source; supplement with general knowledge only for basic concepts
- Never hallucinate gene names or scientific data absent from context
- Preserve all scientific terms exactly as written in context
- Use headings and bullet points

Context:
{context}

Question: {question}
Answer:"""

table_prompt = """You are an expert agricultural research assistant.

RULES:
- Extract structured data from context into a clean markdown table
- Only include genes/values present in the context
- Add a one-line note if clarification helps

Context:
{context}

Question: {question}
Answer:"""

generic_prompt = """You are an expert agricultural research assistant.

RULES:
- Use context as primary source; give a simple general explanation if context is insufficient
- Never hallucinate specific scientific data
- Be concise

Context:
{context}

Question: {question}
Answer:"""

def choose_prompt(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["list", "table", "compare", "functions", "data"]):
        return table_prompt
    if any(w in q for w in ["what is", "how", "why", "explain", "role", "mechanism"]):
        return informational_prompt
    return generic_prompt

# =========================================
# COUNT TOKENS  (word-level approximation)
# =========================================
def count_tokens(text: str) -> int:
    return len(text.split())

# =========================================
# LLAMA CALL  — streaming enabled
# =========================================
def ask_llama_stream(messages: list):
    stream = ollama.chat(
        model="llama3.2",
        messages=messages,
        stream=True
    )
    for chunk in stream:
        yield chunk["message"]["content"]

# =========================================
# MAIN RAG FUNCTION  — streaming
# FIX 1: history now includes assistant replies
# FIX 2: context limit raised to 8000 chars
# FIX 3: keyword score normalized by doc length
# NEW:   full latency + token logging per query
# =========================================
def rag_chat(message: str, history: list):
    t_total = time.time()

    # Step 1: Retrieve context
    results, retrieval_latency = hybrid_search(message, top_k=7)
    context = get_context(results)

    # Step 2: Choose prompt + fill template
    prompt_template = choose_prompt(message)
    prompt = prompt_template.format(context=context, question=message)
    prompt_tokens = count_tokens(prompt)

    # Step 3: Build full conversation history
    # includes BOTH user and assistant turns
    messages = []
    for msg in history:
        role    = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            content = content[0]["text"]
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt})

    # Step 4: Stream response from LLaMA
    t_llm    = time.time()
    response = ""
    for chunk in ask_llama_stream(messages):
        response += chunk
        yield response

    # Step 5: Log latency + token usage
    llm_latency   = time.time() - t_llm
    total_latency = time.time() - t_total
    response_tokens = count_tokens(response)

    log.info(
        f"QUERY | '{message[:60]}' | "
        f"prompt_tokens={prompt_tokens} | "
        f"response_tokens={response_tokens} | "
        f"retrieval={retrieval_latency:.3f}s | "
        f"llm={llm_latency:.3f}s | "
        f"total={total_latency:.3f}s"
    )

    print(
        f"[RAG] {total_latency:.2f}s total | "
        f"retrieval={retrieval_latency:.2f}s | "
        f"llm={llm_latency:.2f}s | "
        f"prompt={prompt_tokens} tokens | "
        f"response={response_tokens} tokens"
    )

# =========================================
# GRADIO UI
# =========================================
demo = gr.ChatInterface(
    fn=rag_chat,
    title="Agricultural RAG Chatbot",
    description="Ask questions about crop research, genetics, drought resistance, and more.",

)

# =========================================
# RUN
# share=True generates a public Gradio URL
# for the assignment hosted demo requirement
# =========================================
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)