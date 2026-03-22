import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# =========================================
# 🔥 STEP 1: LOAD DATA
# =========================================
def load_data():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    return chunks, embeddings


# =========================================
# 🔥 STEP 2: CONNECT QDRANT
# =========================================
client = QdrantClient(host="localhost", port=6333)

collection_name = "agri_rag_final"


# =========================================
# 🔥 STEP 3: CREATE COLLECTION
# =========================================
def create_collection():
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,                 # IMPORTANT (MiniLM)
            distance=Distance.COSINE
        )
    )
    print("✅ Collection created")


# =========================================
# 🔥 STEP 4: PREPARE POINTS
# =========================================
def prepare_points(chunks, embeddings):
    points = []

    for chunk, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=chunk["id"],
                vector=vector.tolist(),
                payload={
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "source": chunk["source"]
                }
            )
        )

    return points


# =========================================
# 🔥 STEP 5: UPLOAD IN BATCHES
# =========================================
def upload_in_batches(points, batch_size=1000):
    total = len(points)

    for i in range(0, total, batch_size):
        batch = points[i:i + batch_size]

        client.upsert(
            collection_name=collection_name,
            points=batch
        )

        print(f"⬆️ Uploaded {i + len(batch)} / {total}")


# =========================================
# 🚀 MAIN PIPELINE
# =========================================
if __name__ == "__main__":
    print("📂 Loading data...")
    chunks, embeddings = load_data()

    print(f"📊 Total chunks: {len(chunks)}")
    print(f"📊 Embeddings shape: {len(embeddings)}")

    print("🗄️ Creating collection...")
    create_collection()

    print("📦 Preparing points...")
    points = prepare_points(chunks, embeddings)

    print("⬆️ Uploading to Qdrant...")
    upload_in_batches(points)

    print("✅ All vectors stored successfully!")