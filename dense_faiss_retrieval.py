import faiss
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -------------------------
# Load model
# -------------------------
print("[1] Loading SentenceTransformer model...")
model = SentenceTransformer("dense_model_finetuned")
print("Model loaded.\n")

# -------------------------
# Load data
# -------------------------

def load_jsonl_corpus(path):
    print(f"[2] Loading corpus from {path} ...")
    corpus = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            did = obj["_id"]
            # Combine title + text (if title missing, handle safely)
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            corpus[did] = text
    print(f"Loaded {len(corpus)} corpus documents.\n")
    return corpus

def load_jsonl_queries(path):
    print(f"[3] Loading queries from {path} ...")
    queries = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["_id"]
            queries[qid] = obj["text"]
    print(f"Loaded {len(queries)} queries.\n")
    return queries


corpus = load_jsonl_corpus("../../data/PAR_clean/sampled_corpus_200k.jsonl")
corpus_ids = list(corpus.keys())
corpus_text = list(corpus.values())

test_queries = load_jsonl_queries("../../data/PAR_clean/test_queries_10k.jsonl")
test_ids = list(test_queries.keys())
test_text = list(test_queries.values())

# Train relevance for expansion
print("[4] Loading qrels for expansion...")
train_qrels = {}
with open("../../data/PAR_clean/80k_qrels_train.tsv") as f:
    next(f)
    for line in f:
        qid, did, rel = line.strip().split("\t")
        if qid not in train_qrels:
            train_qrels[qid] = []
        train_qrels[qid].append(did)
print(f"Loaded qrels for {len(train_qrels)} training queries.\n")

# -------------------------
# Encode embeddings
# -------------------------
print("[5] Encoding corpus embeddings...")
doc_emb = model.encode(corpus_text, convert_to_numpy=True)
print(f"Corpus encoded. Shape = {doc_emb.shape}")

print("[6] Encoding query embeddings...")
query_emb = model.encode(test_text, convert_to_numpy=True)
print(f"Queries encoded. Shape = {query_emb.shape}\n")

print("[7] Normalizing embeddings...")
faiss.normalize_L2(doc_emb)
faiss.normalize_L2(query_emb)
print("Normalization complete.\n")

# -------------------------
# Build index
# -------------------------
print("[8] Building FAISS index...")
dim = doc_emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_emb)
print("Index built and documents added.\n")

# -------------------------
# Retrieval + Expansion
# -------------------------
def expand_neighbors(candidate_dict, doc_id, score, train_qrels):
    """Add neighbors from train relevance (NN-expansion)."""
    if doc_id in train_qrels:
        for n in train_qrels[doc_id]:
            candidate_dict[n] = candidate_dict.get(n, 0) + score


def retrieve_k(qid, q_index, k=1000):
    D, I = index.search(query_emb[q_index : q_index + 1], k)
    candidate = {}
    
    for rank, doc_idx in enumerate(I[0]):
        did = corpus_ids[doc_idx]
        score = D[0][rank]
        
        candidate[did] = candidate.get(did, 0) + score
        expand_neighbors(candidate, did, score, train_qrels)

    # Final sorted ranking
    return {doc: s for doc, s in sorted(candidate.items(), key=lambda x: -x[1])}


# -------------------------
# Produce results dict
# -------------------------
print("[9] Running retrieval + expansion for all queries...\n")
results = {}
for i, qid in tqdm(enumerate(test_ids)):
    results[qid] = retrieve_k(qid, i, k=500)

print("\n[10] Saving results...")
json.dump(results, open("dense_faiss_results.json", "w"), indent=2)
print("Saved results → dense_faiss_results.json")

print("DONE! Dense retrieval + NN expansion completed successfully.")