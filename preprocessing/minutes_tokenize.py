import pandas as pd
import json
import re
import os
from ekonlpy.tag import Mecab
from concurrent.futures import ProcessPoolExecutor
mecab = Mecab()

INPUT_CSV = "minutes_filtered.csv"
OUTPUT_CSV = "minutes_tokens.csv"

df = pd.read_csv(INPUT_CSV)

negation_words = {"않", "못", "아니"}

def tokenize_ekonlpy(text):
    tokens = mecab.pos(text)

    result = []
    for word, pos in tokens:
        # 명사 / 동사 / 형용사 / 부사
        if (
            pos.startswith("N") or   # 명사
            pos.startswith("V") or   # 동사
            pos == "VA" or            # 형용사
            pos == "MAG" or           # 부사
            word in negation_words    # 부정어
        ):
            result.append(word)

    return result



rows = []

for _, row in df.iterrows():
    release_date = row["release_date"]
    sentences = json.loads(row["sentences"])  

    tokenized_sentences = []

    for sent in sentences:
        tokens = tokenize_ekonlpy(sent)
        if tokens:                              # 빈 문장 제거
            tokenized_sentences.append(tokens)

    rows.append({
        "release_date": release_date,
        "tokens": json.dumps(tokenized_sentences, ensure_ascii=False),
        "source_type": "minutes"
    })

df_tokens = pd.DataFrame(rows)

df_tokens.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)





