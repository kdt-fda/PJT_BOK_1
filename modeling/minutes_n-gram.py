import json, csv
import pandas as pd
from tqdm import tqdm
from ekonlpy.sentiment import MPCK
mpck = MPCK()


def ngramize(tokens, max_n=5):
    keep_tags = {'NNG', 'VV', 'VA', 'MAG'}
    filtered = [w for w in tokens if w.split('/')[-1] in keep_tags]

    ngram_results = []
    L = len(filtered)
    for pos in range(L):
        for n in range(1, max_n + 1):
            end = pos + n
            if end <= L:
                ngram_results.append(";".join(filtered[pos:end]))
    return ngram_results


df_src = pd.read_csv("minutes_filtered.csv", encoding="utf-8-sig")

results = []

for _, row in df_src.iterrows():
    try:
        release_date = row["release_date"]
        text = row["sentences"]

        if pd.isna(text) or not str(text).strip():
            continue

        # 형태소 분석
        tokens = mpck.tokenize(text)

        # n-gram 생성
        ngrams = ngramize(tokens, max_n=5)

        results.append({
            "release_date": release_date,
            "tokens": json.dumps(ngrams, ensure_ascii=False),
            "category": "의사록",
        })

    except Exception as e:
        print(f"{release_date} 처리 중 에러: {e}")


df_out = pd.DataFrame(results)

output_path = "minutes_tokenize.csv"
df_out.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_ALL
)
