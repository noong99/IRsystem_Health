# 이미 해서 이 파일 삭제 가능
# import tarfile

# tar_path = "data/ReCDS_benchmark.tar"

# # tar 파일 열기
# with tarfile.open(tar_path, "r") as tar:
#     # 포함된 모든 파일 이름 출력
#     for member in tar.getmembers():
#         print(member.name)

# # Extract all files
# extract_path = "data/ReCDS_benchmark"  # 새 폴더

# with tarfile.open(tar_path, "r") as tar:
#     tar.extractall(path=extract_path)

# print("✅ Files extracted to:", extract_path)

# # Corpus.jsonl 확인
# import json

# file_path = "data/PAR/corpus.jsonl"

# with open(file_path, "r") as f:
#     first_line = f.readline().strip()  # 첫 번째 줄 읽기
#     sample = json.loads(first_line)    # JSON 파싱

# print(sample.keys())

# # relevance score 확인
# import pandas as pd

# # 탭(\t) 구분자 사용
# qrels = pd.read_csv("data/PAR/qrels_train.tsv", sep="\t")

# # 데이터 구조 확인
# print(qrels.head())

# # relevance 컬럼의 고유한 값(라벨 종류)
# print(qrels["score"].unique())


# #  # PMC-Patients-V2.json 확인
# import json

# # 파일 열기
# with open('data/PMC-Patients-V2.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 예시: 첫 번째 항목 보기
# print(data[0])

# #  # PMC-Patients-V2.json 확인
# import json

# # 파일 열기
# with open('data/PAR/corpus.jsonl', 'r', encoding='utf-8') as f:
#     data = [json.loads(line) for line in f if line.strip()]

# print(data[0])

# # 우리의 own corpus 확인
# import json

# file_path = "data/PAR_clean/sampled_corpus_200k.jsonl"

# with open(file_path, "r") as f:
#     data = [json.loads(line) for line in f if line.strip()]

# print(data[0])

# # irrelevant.jsonl 확인
# import json

# def read_jsonl(file_path):
#     """Read JSONL file and return list of documents"""
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 data.append(json.loads(line))
#     return data

# # Read irrelevant.jsonl
# file_path = 'data/PAR/irrelevant.jsonl'
# data = read_jsonl(file_path)

# # Print length and first document as sample
# print(f"Number of documents in irrelevant.jsonl: {len(data)}")
# if data:
#     print("\nSample document (first one):")
#     print(json.dumps(data[0], indent=2, ensure_ascii=False))

# irrelevant 파일 확인
# import json
# count = 0
# with open("data/PAR/irrelevant.jsonl", "r") as f:
#     for _ in f:
#         count += 1

# print(f"Total entries in irrelevant.jsonl: {count}") # 10,789,853

# import json

# with open("data/PAR/irrelevant.jsonl", "r") as f:
#     first_line = f.readline().strip()   # 첫 줄 읽기 (문자열)
#     first_entry = json.loads(first_line)  # JSON 객체로 변환

# print(first_entry)

# Count queries in train, dev, test JSONL files
def count_jsonl_lines(filepath):
    count = 0
    with open(filepath, "r") as f:
        for _ in f:
            count += 1
    return count

train_queries = count_jsonl_lines("data/queries/train_queries.jsonl")
dev_queries = count_jsonl_lines("data/queries/dev_queries.jsonl")
test_queries = count_jsonl_lines("data/queries/test_queries.jsonl")

total_queries = train_queries + dev_queries + test_queries

print(f"# Train queries: {train_queries}") # 155151
print(f"# Dev queries: {dev_queries}")  # 5924
print(f"# Test queries: {test_queries}") # 5959
print(f"Total queries: {total_queries}") # 167034
