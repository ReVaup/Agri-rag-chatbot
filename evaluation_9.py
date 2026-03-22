import os
import time
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =========================================
# LOGGING
# =========================================
logging.basicConfig(
    filename="eval_results.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)
log = logging.getLogger(__name__)

# =========================================
# SETUP
# =========================================
import torch
device          = "cuda" if torch.cuda.is_available() else "cpu"
model           = SentenceTransformer("all-MiniLM-L6-v2", device=device)
client          = QdrantClient(host="localhost", port=6333)
collection_name = "agri_rag_final"

# =========================================
# 20 EVALUATION QUERIES
# Each has:
#   "query"           — the question
#   "expected_source" — substring of the paper filename that should appear
#                       in top-5 results (used for precision@5)
#   "expected_terms"  — key scientific terms that MUST appear in retrieved
#                       text if retrieval is correct (used for recall check)
# =========================================
EVAL_QUERIES = [
    {
        "query": "What genes are responsible for pod shattering resistance in soybean?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["qPHD1", "SHAT1-5"]
    },
    {
        "query": "What is the role of SiPRX genes in sesame under drought stress?",
        "expected_source": "Plant Stress_11_1-13",
        "expected_terms": ["SiPRX", "peroxidase", "drought"]
    },
    {
        "query": "How does chickpea domestication relate to the Fertile Crescent?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["chickpea", "Fertile Crescent", "domestication"]
    },
    {
        "query": "What QTLs are associated with root-knot nematode resistance in rice?",
        "expected_source": "Frontiers in Genetics_13_1-15",
        "expected_terms": ["QTL", "nematode", "rice"]
    },
    {
        "query": "What is the genome size of soybean and key resistance genes?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["soybean", "genome", "Glycine max"]
    },
    {
        "query": "What are the key descriptors and nutritional traits of foxtail millet Setaria italica?",
        "expected_source": "Key descriptors for foxtail millet",
        "expected_terms": ["millet", "nutritional"]
    },
    {
        "query": "What are the key domestication syndrome traits in food legumes?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["domestication syndrome", "legume", "seed"]
    },
    {
        "query": "What genes control flowering time variation in common bean?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["flowering", "common bean", "Phaseolus"]
    },
    {
        "query": "What is the role of WRKY family genes in stress tolerance in Cajanus cajan pigeon pea?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["WRKY", "pigeon pea", "stress"]
    },
    {
        "query": "How is marker-assisted selection used in legume domestication and crop improvement QTL mapping?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["marker", "selection", "QTL"]
    },
    {
        "query": "What is BSA-seq and how is it used in plant genetics?",
        "expected_source": "Frontiers in Genetics_13_1-15",
        "expected_terms": ["BSA", "bulk segregant", "sequencing"]
    },
    {
        "query": "What are the nutritional benefits of millets for human health?",
        "expected_source": "Nutrients_14_1-16",
        "expected_terms": ["millet", "nutrition", "protein"]
    },
    {
        "query": "How does drought stress affect SiPRX gene expression in sesame crops?",
        "expected_source": "Plant Stress_11_1-13",
        "expected_terms": ["drought", "stress", "expression"]
    },
    {
        "query": "What is the origin of pigeon pea and where was it first domesticated?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["pigeon pea", "Cajanus", "domesticated"]
    },
    {
        "query": "What is genome editing and how is it used in crop domestication?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["genome editing", "CRISPR", "domestication"]
    },
    {
        "query": "What are the major subfamilies of the Leguminosae family?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["Papilionoideae", "Caesalpinioideae", "Leguminosae"]
    },
    {
        "query": "How does seed dormancy relate to crop domestication in legumes?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["seed dormancy", "domestication", "legume"]
    },
    {
        "query": "What is the role of nitrogen fixation in legume crops?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["nitrogen", "fixation", "rhizobia"]
    },
    {
        "query": "What is the function of CcHsfA-1d and CcHsfA-2 genes in heat stress response in pigeon pea?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["heat", "CcHsfA", "pigeon pea"]
    },
    {
        "query": "What is the significance of whole genome duplication in legume evolution?",
        "expected_source": "Frontiers in Genetics_13_1-21",
        "expected_terms": ["genome duplication", "WGD", "legume"]
    },
]

# =========================================
# EMBED + SEARCH
# =========================================
def embed_query(query: str):
    return model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

def keyword_score(query: str, text: str) -> float:
    query_words = set(query.lower().split())
    text_words  = text.lower().split()
    if not text_words:
        return 0.0
    return sum(1 for w in text_words if w in query_words) / len(text_words)

def hybrid_search(query: str, top_k: int = 5):
    vec     = embed_query(query)
    results = client.query_points(
        collection_name=collection_name,
        query=vec,
        limit=10
    ).points

    scored = []
    for r in results:
        text       = r.payload.get("text", "")
        chunk_type = r.payload.get("type", "text")
        sem        = r.score
        kw         = keyword_score(query, text)
        boost      = 0.05 if chunk_type == "table" else 0.0
        scored.append((sem + 0.1 * kw + boost, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]

# =========================================
# EVALUATION METRICS
# =========================================
def check_precision(results, expected_source: str) -> bool:
    """Returns True if expected source appears in any of the top-5 results."""
    for r in results:
        source = r.payload.get("source", "")
        if expected_source.lower() in source.lower():
            return True
    return False

def check_recall(results, expected_terms: list) -> dict:
    """Returns which expected terms appear in any retrieved chunk."""
    all_text = " ".join(r.payload.get("text", "") for r in results).lower()
    return {
        term: term.lower() in all_text
        for term in expected_terms
    }

def check_latency(query: str) -> float:
    """Returns retrieval latency in seconds."""
    t0 = time.time()
    hybrid_search(query)
    return time.time() - t0

# =========================================
# RUN EVALUATION
# =========================================
def run_evaluation():
    print("=" * 65)
    print("RAG EVALUATION — 20 queries, precision@5 + recall + latency")
    print("=" * 65)

    precision_hits  = 0
    latencies       = []
    recall_scores   = []
    results_detail  = []

    for i, item in enumerate(EVAL_QUERIES):
        query           = item["query"]
        expected_source = item["expected_source"]
        expected_terms  = item["expected_terms"]

        # Time the retrieval
        t0      = time.time()
        results = hybrid_search(query, top_k=5)
        latency = time.time() - t0
        latencies.append(latency)

        # Precision@5
        hit = check_precision(results, expected_source)
        if hit:
            precision_hits += 1

        # Recall — term coverage
        term_hits   = check_recall(results, expected_terms)
        recall_pct  = sum(term_hits.values()) / len(term_hits) * 100
        recall_scores.append(recall_pct)

        # Top sources retrieved
        top_sources = [r.payload.get("source", "")[:45] for r in results[:3]]

        status = "PASS" if hit else "FAIL"
        print(f"\nQ{i+1:02d} [{status}] {query[:60]}")
        print(f"     Expected : {expected_source[:50]}")
        print(f"     Retrieved: {top_sources[0]}")
        print(f"     Latency  : {latency:.3f}s  |  Term recall: {recall_pct:.0f}%  {term_hits}")

        log.info(
            f"Q{i+1:02d} | {status} | latency={latency:.3f}s | recall={recall_pct:.0f}% | "
            f"query='{query[:50]}' | expected='{expected_source[:40]}'"
        )

        results_detail.append({
            "query":     query,
            "hit":       hit,
            "latency":   latency,
            "recall":    recall_pct,
            "term_hits": term_hits
        })

    # ── Summary ───────────────────────────────────────────────────
    precision_at_5   = precision_hits / len(EVAL_QUERIES) * 100
    avg_latency      = np.mean(latencies)
    p95_latency      = np.percentile(latencies, 95)
    avg_recall       = np.mean(recall_scores)
    under_5s         = sum(1 for l in latencies if l < 5.0)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Precision@5          : {precision_at_5:.1f}%  ({precision_hits}/{len(EVAL_QUERIES)} queries)")
    print(f"Avg term recall      : {avg_recall:.1f}%")
    print(f"Avg retrieval latency: {avg_latency:.3f}s")
    print(f"P95 retrieval latency: {p95_latency:.3f}s")
    print(f"Queries under 5s     : {under_5s}/{len(EVAL_QUERIES)}")
    print(f"Min latency          : {min(latencies):.3f}s")
    print(f"Max latency          : {max(latencies):.3f}s")

    # Assignment targets
    print("\nAssignment targets:")
    print(f"  >80% precision@5 : {'PASS' if precision_at_5 >= 80 else 'FAIL'}  ({precision_at_5:.1f}%)")
    print(f"  <5s latency      : {'PASS' if avg_latency < 5.0 else 'FAIL'}  ({avg_latency:.3f}s avg)")

    log.info(
        f"SUMMARY | precision@5={precision_at_5:.1f}% | "
        f"avg_recall={avg_recall:.1f}% | "
        f"avg_latency={avg_latency:.3f}s | "
        f"p95_latency={p95_latency:.3f}s"
    )

    return results_detail

if __name__ == "__main__":
    run_evaluation()