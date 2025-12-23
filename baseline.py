import json
import math
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import re
import pandas as pd
import pickle
import os

# ============================================================
# 1. Tokenizer (RegexTokenizer)
# ============================================================

class RegexTokenizer:
    def __init__(self, pattern=r"\b\w+\b"):
        self.pattern = re.compile(pattern)

    def tokenize(self, text):
        return [t.lower() for t in self.pattern.findall(text)]

# ============================================================
# 2. Basic Inverted Index with BM25
# ============================================================

class BasicInvertedIndex:
    def __init__(self, k1=1.2, b=0.75):
        self.index = defaultdict(list)
        self.doc_lengths = {}
        self.N = 0
        self.k1 = k1
        self.b = b
        self.avgdl = 0.0

    def add_document(self, doc_id, tokens):
        self.N += 1
        self.doc_lengths[doc_id] = len(tokens)
        term_freq = Counter(tokens)
        for term, freq in term_freq.items():
            self.index[term].append((doc_id, freq))

    def finalize(self):
        self.avgdl = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def bm25_score(self, query_tokens):
        scores = defaultdict(float)
        for term in query_tokens:
            postings = self.index.get(term, [])
            n = len(postings)
            if n == 0:
                continue
            idf = math.log((self.N - n + 0.5) / (n + 0.5) + 1)
            for doc_id, freq in postings:
                dl = self.doc_lengths[doc_id]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * freq * (self.k1 + 1) / denom
                scores[doc_id] += score
        return scores

# ============================================================
# 3. Load PMC-Patients (ReCDS format)
# ============================================================

TRAIN_Q_PATH = "data/queries/train_queries.jsonl"
TEST_Q_PATH  = "data/queries/test_queries.jsonl"
CORPUS_PATH  = "data/PAR/corpus.jsonl"   
QRELS_TRAIN_PATH = "data/PAR/qrels_train.tsv"
QRELS_TEST_PATH  = "data/PAR/qrels_test.tsv"
PMID_PATH = "data/PAR_PMIDs.json"
INDEX_PATH = "inverted_index.pkl"   # 저장 경로

# ---- Load queries (train + test)
queries = {}
for path in [TRAIN_Q_PATH, TEST_Q_PATH]:
    with open(path, "r") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

# ---- Load corpus
corpus = {}
with open(CORPUS_PATH, "r") as f:
    for line in f:
        d = json.loads(line)
        corpus[str(d["_id"])] = d["text"]

# ---- Load qrels (relevance)
def load_qrels_tsv(path):
    qrels = defaultdict(dict)
    df = pd.read_csv(path, sep="\t", header=0)
    for _, row in df.iterrows():
        qid = str(row.iloc[0])
        docid = str(row.iloc[1])
        rel = int(row.iloc[2])
        qrels[qid][docid] = rel
    return qrels

qrels = load_qrels_tsv(QRELS_TRAIN_PATH)
qrels_test = load_qrels_tsv(QRELS_TEST_PATH)
qrels.update(qrels_test)

print(f"Loaded {len(queries)} queries, {len(corpus)} documents, {len(qrels)} qrels.")

# ============================================================
# 4. Indexing (Load if exists, else build)
# ============================================================

tokenizer = RegexTokenizer()

if os.path.exists(INDEX_PATH):
    print(f"Found existing index file → Loading from {INDEX_PATH} ...")
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    print("Inverted index loaded successfully!")
else:
    print("No existing index found → Building new inverted index...")
    index = BasicInvertedIndex()
    for doc_id, text in tqdm(corpus.items(), desc="Indexing"):
        tokens = tokenizer.tokenize(text)
        index.add_document(doc_id, tokens)
    index.finalize()

    # Save index
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"Inverted index built and saved as {INDEX_PATH}!")

# ============================================================
# 5. Retrieval
# ============================================================

results = {}

for qid, query_text in tqdm(queries.items(), desc="Retrieving"):
    query_tokens = tokenizer.tokenize(query_text)
    scores = index.bm25_score(query_tokens)
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
    results[qid] = ranked_docs

print(f"Retrieved results for {len(results)} queries.")

# ============================================================
# 6. Evaluation (nDCG@10 + MAP)
# ============================================================

def dcg(rels):
    return np.sum([(2**rel - 1) / np.log2(i+2) for i, rel in enumerate(rels)])

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

print(f"Mean nDCG@10 = {np.mean(ndcg_scores):.4f}")
print(f"Mean MAP = {np.mean(map_scores):.4f}")
