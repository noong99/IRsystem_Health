import json
import pandas as pd
from pathlib import Path

# 파일 경로 설정
corpus_path = "data/PAR/corpus.jsonl"
qrels_paths = [
    "data/PAR/qrels_dev.tsv",
    "data/PAR/qrels_test.tsv",
    "data/PAR/qrels_train.tsv"
]
output_path = "data/PAR/irrelevant.jsonl"

def read_jsonl(file_path):
    """JSONL 파일을 한 줄씩 읽어서 리스트로 반환"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_all_relevant_ids():
    """모든 qrels 파일에서 corpus-id를 수집"""
    relevant_ids = set()
    
    for qrels_path in qrels_paths:
        # TSV 파일 읽기 (탭으로 구분)
        df = pd.read_csv(qrels_path, sep='\t')
        # corpus-id 컬럼의 모든 값을 set에 추가
        relevant_ids.update(df['corpus-id'].unique())
    
    return relevant_ids

def main():
    # corpus.jsonl 파일 읽기
    print("Reading corpus.jsonl...")
    corpus_data = read_jsonl(corpus_path)
    print(f"Total documents in corpus: {len(corpus_data)}")
    
    # corpus의 모든 ID 수집
    corpus_ids = {str(doc['_id']) for doc in corpus_data}  # ID를 문자열로 변환
    print(f"Unique IDs in corpus: {len(corpus_ids)}")
    
    # qrels 파일들에서 모든 relevant ID 수집
    print("Collecting relevant IDs from qrels files...")
    relevant_ids = get_all_relevant_ids()
    print(f"Total relevant IDs: {len(relevant_ids)}")
    
    # relevant_ids에 없는 문서만 찾기
    # corpus_ids와 relevant_ids의 차집합을 구함
    irrelevant_ids = corpus_ids - {str(id) for id in relevant_ids}  # ID를 문자열로 변환
    print(f"Number of irrelevant IDs: {len(irrelevant_ids)}")
    
    # 매핑되지 않는 문서 찾기
    irrelevant_docs = [doc for doc in corpus_data if str(doc['_id']) in irrelevant_ids]
    print(f"Found {len(irrelevant_docs)} irrelevant documents")
    
    # irrelevant.jsonl 파일로 저장
    print(f"Saving irrelevant documents to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in irrelevant_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print("✅ Done!")
    print(f"Saved {len(irrelevant_docs)} documents to {output_path}")

if __name__ == "__main__":
    main()