import json
import numpy as np

import sys
import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../BM25_Baseline/"))
sys.path.append(base_path)

from relevance import map_score, ndcg_score

# -------------------------
# Load qrels
# -------------------------
def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)
        for line in f:
            qid, did, rel = line.strip().split("\t")
            qrels.setdefault(qid, {})[did] = int(rel)
    return qrels

# -------------------------
# Eval MAP/NDCG
# -------------------------
def eval_dense(results, qrels, k=10):
    map_list, ndcg_list = [], []

    for qid, rels in qrels.items():
        if qid not in results:
            map_list.append(0)
            ndcg_list.append(0)
            continue

        ranked_docs = list(results[qid].keys())[:k]

        bin_rels = [1 if rels.get(d, 0) > 0 else 0 for d in ranked_docs]
        grad = [float(rels.get(d, 0)) for d in ranked_docs]
        ideal = sorted(rels.values(), reverse=True)

        map_list.append(map_score(bin_rels, k))
        ndcg_list.append(ndcg_score(grad, ideal, k))

    return map_list, ndcg_list

def mean_ci(a):
    a = np.array(a)
    mean = a.mean()
    ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    return mean, ci


qrels = load_qrels("../../data/PAR_clean/10k_qrels_test.tsv") # 2000으로 할거면 10k_qrels_test_to_2000.tsv
results = json.load(open("dense_faiss_results.json"))

map_list, ndcg_list = eval_dense(results, qrels, k=10)

m_mean, m_ci = mean_ci(map_list)
n_mean, n_ci = mean_ci(ndcg_list)

print(f"MAP@10 = {m_mean:.4f} ± {m_ci:.4f}")
print(f"NDCG@10 = {n_mean:.4f} ± {n_ci:.4f}")

# With 10k qrels test set:
# MAP@10 = 0.2228 ± 0.0093
# NDCG@10 = 0.1965 ± 0.0079

# With first 2000 qrels test set:
# MAP@10 = 0.2142 ± 0.0246
# NDCG@10 = 0.1796 ± 0.0203