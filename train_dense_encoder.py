# 이거는 gpu로 Great Lakes에서 이미 돌림
import json
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm


# -------------------------
# Load train queries (JSONL)
# Format: {"_id": "...", "text": "..."}
# -------------------------
def load_jsonl_queries(path):
    queries = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["_id"]
            queries[qid] = obj["text"]
    return queries


# -------------------------
# Load corpus (JSONL)
# Format: {"_id": "...", "title": "...", "text": "..."}
# -------------------------
def load_jsonl_corpus(path):
    corpus = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            did = obj["_id"]
            # Combine title + text (if title missing, handle safely)
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            corpus[did] = text
    return corpus

# -------------------------
# Load qrels_train.tsv
# -------------------------
def load_qrels(path):
    qrels = {}
    with open(path, "r") as f:
        next(f)   # skip header
        for line in f:
            qid, did, rel = line.strip().split("\t")
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = int(rel)
    return qrels

# -------------------------
# Build training pairs
# -------------------------
def build_training_examples(train_queries, corpus, qrels):
    examples = []

    for qid, rels in tqdm(qrels.items(), desc="Building training pairs"):
        if qid not in train_queries:
            continue  # skip missing

        query_text = train_queries[qid]

        # positive docs
        pos_docs = [d for d, r in rels.items() if r > 0]

        for pos in pos_docs:
            if pos not in corpus:
                continue  # skip missing docs

            pos_text = corpus[pos]

            # Negative sample (random)
            neg = random.choice(list(corpus.keys()))
            neg_text = corpus[neg]

            # For MNR Loss, we only include (query, positive)
            examples.append(InputExample(texts=[query_text, pos_text]))

    return examples


# -------------------------
# Main Training Pipeline
# -------------------------
def main():

    print("Loading data...")
    train_queries = load_jsonl_queries("../../data/PAR_clean/train_queries_80k.jsonl")
    corpus = load_jsonl_corpus("../../data/PAR_clean/sampled_corpus_200k.jsonl")
    qrels = load_qrels("../../data/PAR_clean/80k_qrels_train.tsv")

    print(f"Loaded train queries: {len(train_queries)}")
    print(f"Loaded corpus docs:  {len(corpus)}")
    print(f"Loaded qrels entries: {len(qrels)}")

    # Build training examples
    examples = build_training_examples(train_queries, corpus, qrels)
    print(f"Total training pairs: {len(examples)}")

    # Load model
    model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

    # DataLoader
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=32)

    # MNR Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    print("Training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=1000,
        show_progress_bar=True
    )

    # Save
    model.save("dense_model_finetuned")
    print("Model saved to: dense_model_finetuned/")


if __name__ == "__main__":
    main()