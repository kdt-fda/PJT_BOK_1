import pandas as pd
import re
from ekonlpy.tag import Mecab

mecab = Mecab()

df = pd.read_csv("./preprocessing/minutes_stage1.csv", encoding="utf-8-sig")

# BODY만 추출
df_body = df[df["text_type"] == "BODY"].copy()

print(df_body.shape)
print(df_body.head())

df_body["text_raw"] = df_body["text"].astype(str).fillna("").str.strip()
df_sent = (
    df_body
    .assign(sentence=df_body["text_raw"].astype(str).str.split("음"))
    .explode("sentence")
    .reset_index(drop=True)
)

df_sent["sentence"] = df_sent["sentence"].str.strip()
df_sent = df_sent[df_sent["sentence"] != ""]

# 클렌징 작업 후 csv 저장
SAVE_PATH = "./preprocessing/minutes_filtered.csv"

cols_to_save = [
    "meeting_date",
    "release_date",
    "text_type",
    "sentence"
]

df_sent[cols_to_save].to_csv(
    SAVE_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(df_sent[cols_to_save].head())


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

df_sent["tokens"] = df_sent["sentence"].apply(tokenize_ekonlpy)

print(df_sent[["meeting_date", "release_date", "sentence", "tokens"]].head())



