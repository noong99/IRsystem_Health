"""Compute rough statistics for PAR_clean dataset.

Outputs:
- Number of documents (from PAR_clean/sampled_corpus_200k.jsonl)
- Average document length (whitespace token count of 'text')
- Number of queries (unique query-id present in PAR_clean qrels)
- Average query length (whitespace token count of 'text' from data/queries/*)
- Relevance counts for score==1 and score==2 across PAR_clean qrels

Usage:
  python data_rough_statistics.py --dataset-dir data/PAR_clean --queries-dir data/queries
"""
################# Rough Statistics for PAR_clean Dataset #################
# # Statistic	Value
# Queries	105,623
# Documents	200,000
# Documents with Relevance = 0	101,313
# Documents with Relevance = 1 or 2	98,687
# Qrels rows with score = 1	200,006
# Qrels rows with score = 2	7,870
# Total qrels judgments	207,876
########################################################################

import argparse
import json
import os
from typing import Dict, Iterable, Set, Tuple


def read_qrels_stats(qrels_dir: str) -> Tuple[Set[str], Set[str], Set[str], Set[str], int, int, int]:
    """Read all *.tsv qrels in qrels_dir and compute:
    - used_qids: set of unique query ids
    - docs_with_rel1: set of unique corpus-ids that have at least one score==1
    - docs_with_rel2: set of unique corpus-ids that have at least one score==2
    - relevant_docids: set of unique corpus-ids with score 1 or 2
    - total_rows: total number of qrels rows
    - rel1_count: count of rows with score == 1
    - rel2_count: count of rows with score == 2
    Assumes header row: 'query-id\tcorpus-id\tscore'.
    """
    used_qids: Set[str] = set()
    docs_with_rel1: Set[str] = set()
    docs_with_rel2: Set[str] = set()
    relevant_docids: Set[str] = set()
    total_rows = 0
    rel1 = 0
    rel2 = 0
    for fname in os.listdir(qrels_dir):
        if not fname.endswith('.tsv'):
            continue
        path = os.path.join(qrels_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            header_seen = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not header_seen:
                    header_seen = True
                    # best-effort: skip if this looks like header
                    if line.lower().startswith('query-id'):
                        continue
                parts = line.split('\t')
                if len(parts) < 3:
                    # robust against spaces-only delim
                    parts = line.split()
                if len(parts) < 3:
                    continue
                qid, docid, score_str = parts[0], parts[1], parts[2]
                used_qids.add(qid)
                total_rows += 1
                try:
                    s = int(score_str)
                    if s == 1:
                        rel1 += 1
                        docs_with_rel1.add(docid)
                        relevant_docids.add(docid)
                    elif s == 2:
                        rel2 += 1
                        docs_with_rel2.add(docid)
                        relevant_docids.add(docid)
                except ValueError:
                    # ignore non-integer scores
                    pass
    return used_qids, docs_with_rel1, docs_with_rel2, relevant_docids, total_rows, rel1, rel2


def doc_stats(corpus_jsonl: str) -> Tuple[int, float]:
    """Return (num_docs, avg_doc_len_tokens) by streaming corpus_jsonl.
    Expects each line to be a JSON object containing 'text' (and possibly 'title').
    """
    num_docs = 0
    total_tokens = 0
    with open(corpus_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get('text') or ''
            # simple whitespace tokenization
            tokens = text.split()
            total_tokens += len(tokens)
            num_docs += 1
    avg_len = (total_tokens / num_docs) if num_docs else 0.0
    return num_docs, avg_len


def query_stats(queries_dir: str, used_qids: Set[str]) -> Tuple[int, float, int]:
    """Return (num_queries, avg_query_len_tokens, missing_in_queries)
    by reading train/dev/test queries JSONL in queries_dir and
    selecting only those with _id in used_qids.
    """
    files = ['train_queries.jsonl', 'dev_queries.jsonl', 'test_queries.jsonl']
    found_qids: Set[str] = set()
    total_len = 0
    for fname in files:
        path = os.path.join(queries_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                qid = str(obj.get('_id') or obj.get('id') or '')
                if not qid or qid not in used_qids or qid in found_qids:
                    continue
                text = obj.get('text') or obj.get('query') or ''
                total_len += len(text.split())
                found_qids.add(qid)
    missing = len(used_qids - found_qids)
    num = len(found_qids)
    avg_len = (total_len / num) if num else 0.0
    return num, avg_len, missing


def main():
    parser = argparse.ArgumentParser(description='Compute rough statistics for PAR_clean dataset')
    parser.add_argument('--dataset-dir', default='data/PAR_clean', help='Directory with PAR_clean qrels and corpus')
    parser.add_argument('--queries-dir', default='data/queries', help='Directory with train/dev/test queries jsonl')
    parser.add_argument('--corpus-file', default='sampled_corpus_200k.jsonl', help='Corpus JSONL filename under dataset-dir')
    args = parser.parse_args()

    qrels_dir = args.dataset_dir
    corpus_path = os.path.join(args.dataset_dir, args.corpus_file)

    # 1) Qrels-based stats and query-id set
    # Read qrels TSV files (10k_qrels_dev/test.tsv, 80k_qrels_train.tsv) to:
    #   - collect used query IDs
    #   - collect unique docs with score 1, score 2, and all relevant (1 or 2)
    #   - count relevance scores (1/2)
    used_qids, docs_with_rel1, docs_with_rel2, relevant_docids, total_rows, rel1, rel2 = read_qrels_stats(qrels_dir)

    # 2) Document stats from corpus
    # Count documents and compute average length from sampled_corpus_200k.jsonl
    # (This file = irrelevant docs + all docs referenced in qrels)
    num_docs, avg_doc_len = doc_stats(corpus_path)

    # 3) Compute document relevance breakdown
    # Docs in corpus but NOT in qrels = irrelevant (score 0)
    num_relevant = len(relevant_docids)
    num_irrelevant = num_docs - num_relevant
    
    # Breakdown of relevant docs: only-1, only-2, both-1-and-2
    docs_only_rel1 = docs_with_rel1 - docs_with_rel2  # has score=1 but never score=2
    docs_only_rel2 = docs_with_rel2 - docs_with_rel1  # has score=2 but never score=1
    docs_both = docs_with_rel1 & docs_with_rel2        # has both score=1 and score=2

    # 4) Query stats restricted to used_qids
    num_queries, avg_query_len, missing_queries = query_stats(args.queries_dir, used_qids)

    print('=== PAR_clean statistics ===')
    print(f'Qrels files directory: {qrels_dir}')
    print(f'Corpus file: {corpus_path}')
    print('--- Documents ---')
    print(f'Number of documents: {num_docs:,}')
    print(f'  - Irrelevant (not in qrels, score 0): {num_irrelevant:,}')
    print(f'  - Relevant (in qrels): {num_relevant:,}')
    print(f'    • Only score=1: {len(docs_only_rel1):,}')
    print(f'    • Only score=2: {len(docs_only_rel2):,}')
    print(f'    • Both score=1 and 2: {len(docs_both):,}')
    print(f'Average document length (tokens): {avg_doc_len:.2f}')
    print('--- Queries (only those referenced by PAR_clean qrels) ---')
    print(f'Number of queries: {num_queries:,} (missing from queries files: {missing_queries})')
    print(f'Average query length (tokens): {avg_query_len:.2f}')
    print('--- Qrels Judgments (query-document pairs) ---')
    print(f'Total qrels rows: {total_rows:,}')
    print(f'  - Score=1 judgments: {rel1:,}')
    print(f'  - Score=2 judgments: {rel2:,}')



if __name__ == '__main__':
    main()
