# Compare to the baseline BM25 implementation, 
# calculate using semantic similarity (cosine) instead of 
# term frequency

# ============================================================
# BioBERT Dense Retrieval for PMC-Patients (ReCDS)
# ============================================================

import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ============================================================
# 1. Load the dataset
# ============================================================

TRAIN_Q_PATH = "data/queries/train_queries.jsonl"
TEST_Q_PATH  = "data/queries/test_queries.jsonl"
CORPUS_PATH  = "data/PAR/corpus.jsonl"   
QRELS_TRAIN_PATH = "data/PAR/qrels_train.tsv"
QRELS_TEST_PATH  = "data/PAR/qrels_test.tsv"
PMID_PATH = "data/PAR_PMIDs.json"

# ---- Load queries
queries = {}
for path in [TRAIN_Q_PATH, TEST_Q_PATH]:
    with open(path, "r") as f:
        for line in f:
            q = json.loads(line)
            queries[q["query_id"]] = q["text"]

# ---- Load corpus
corpus = {}
with open(CORPUS_PATH, "r") as f:
    for line in f:
        d = json.loads(line)
        corpus[str(d["doc_id"])] = d["text"]

# ---- Load qrels
def load_qrels_tsv(path):
    qrels = defaultdict(dict)
    df = pd.read_csv(path, sep="\t", header=None)
    for _, row in df.iterrows():
        qid, docid, rel = str(row[0]), str(row[1]), int(row[2])
        qrels[qid][docid] = rel
    return qrels

qrels = load_qrels_tsv(QRELS_TRAIN_PATH)
qrels_test = load_qrels_tsv(QRELS_TEST_PATH)
qrels.update(qrels_test)

print(f"Loaded {len(queries)} queries, {len(corpus)} documents, {len(qrels)} qrels.")

# ============================================================
# 2. Load BioBERT model for embeddings
# ============================================================

# BioBERT (huggingface hub: dmis-lab/biobert-base-cased-v1.1)
# You can use any biomedical SentenceTransformer variant, e.g., BioSentVec, PubMedBERT
model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
model = SentenceTransformer(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}: {model_name}")

# ============================================================
# 3. Create document embeddings (indexing step)
# ============================================================

doc_ids = list(corpus.keys())  # 환자 id....?이면 PAR_PMIDs.json 참고
doc_texts = [corpus[did] for did in doc_ids]

print("Encoding corpus embeddings (this may take a while)...")
doc_embeddings = model.encode(
    doc_texts,
    convert_to_tensor=True,
    show_progress_bar=True,
    batch_size=16
)

torch.save({"ids": doc_ids, "embeddings": doc_embeddings}, "corpus_embeddings.pt")
print("Document embeddings saved successfully.")

# ============================================================
# 4. Retrieval for each query
# ============================================================

corpus_data = torch.load("corpus_embeddings.pt")
doc_ids = corpus_data["ids"]
doc_embeddings = corpus_data["embeddings"]

results = {}

for qid, qtext in tqdm(queries.items(), desc="Retrieving"):
    query_emb = model.encode(qtext, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, doc_embeddings)[0]

    top_k = 100
    top_results = torch.topk(cos_scores, k=top_k)
    top_doc_indices = top_results.indices.cpu().numpy()
    top_doc_scores = top_results.values.cpu().numpy()

    ranked_docs = [(doc_ids[i], float(top_doc_scores[j])) for j, i in enumerate(top_doc_indices)]
    results[qid] = ranked_docs

torch.save(results, "biobert_results.pt")
print("Dense retrieval complete and results saved.")

# ============================================================
# 5. Evaluation (nDCG@10 + MAP)
# ============================================================

def dcg(rels):
    return np.sum([(2**rel - 1) / np.log2(i+1) for i, rel in enumerate(rels)])

def ndcg_at_k(pred_docs, true_rels, k=10):
    rels = [true_rels.get(doc_id, 0) for doc_id, _ in pred_docs[:k]]
    ideal = sorted(true_rels.values(), reverse=True)[:k]
    return dcg(rels) / (dcg(ideal) + 1e-9)

def average_precision(pred_docs, true_rels):
    num_rel = 0
    precision_sum = 0
    for i, (doc_id, _) in enumerate(pred_docs):
        if true_rels.get(doc_id, 0) > 0:
            num_rel += 1
            precision_sum += num_rel / (i + 1)
    total_rel = sum(1 for r in true_rels.values() if r > 0)
    return precision_sum / (total_rel + 1e-9)

ndcg_scores, map_scores = [], []

for qid, ranked_docs in results.items():
    true_rels = qrels.get(qid, {})
    if len(true_rels) > 0:
        ndcg_scores.append(ndcg_at_k(ranked_docs, true_rels, k=10))
        map_scores.append(average_precision(ranked_docs, true_rels))

mean_ndcg = np.mean(ndcg_scores)
mean_map = np.mean(map_scores)
print(f"Mean nDCG@10 = {mean_ndcg:.4f}")
print(f"Mean MAP = {mean_map:.4f}")


