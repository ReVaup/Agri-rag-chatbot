import torch
import pickle
from sentence_transformers import SentenceTransformer

# =========================================
# 🔥 STEP 1: CHECK GPU
# =========================================
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available. Fix PyTorch installation.")

print("✅ Using GPU:", torch.cuda.get_device_name(0))


# =========================================
# 🔥 STEP 2: LOAD MODEL
# =========================================
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cuda"
)


# =========================================
# 🔥 STEP 3: LOAD CHUNKS (FROM FILE)
# =========================================
def load_chunks(file_path="chunks.pkl"):
    with open(file_path, "rb") as f:
        chunks = pickle.load(f)
    return chunks


# =========================================
# 🔥 STEP 4: EMBEDDING FUNCTION
# =========================================
def embed_chunks(chunks, batch_size=128):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# =========================================
# 🔥 STEP 5: SAVE EMBEDDINGS
# =========================================
def save_embeddings(embeddings, file_path="embeddings.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)


# =========================================
# 🚀 MAIN PIPELINE
# =========================================
if __name__ == "__main__":
    print("📂 Loading chunks...")
    chunks = load_chunks()

    print(f"📊 Total chunks: {len(chunks)}")

    print("🚀 Generating embeddings (GPU)...")
    embeddings = embed_chunks(chunks)

    print("💾 Saving embeddings...")
    save_embeddings(embeddings)

    print("✅ Embeddings ready!")
    print("Shape:", embeddings.shape)