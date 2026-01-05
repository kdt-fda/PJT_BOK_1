# 라이브러리
import os # path관리
from ekonlpy import Mecab #ekonlpy 설치
from kss import split_sentences
import re # 정규표현식
import csv
import pandas as pd
from tqdm.auto import tqdm
import time

# 문장 분리하기

path = r"C:\Users\User\Desktop\kdt\pjt_bok_1\debenture_reports_filtered.csv"
df = pd.read_csv(path)
import pandas as pd
from kss import split_sentences

# 1)문장 분리 (NaN 안전)
def split_doc_to_sentences(x):
    if pd.isna(x):
        return []
    t = str(x).strip()
    if not t:
        return []
    sents = split_sentences(t)
    return [s.strip() for s in sents if str(s).strip()]

df["_sent_list"] = df["text"].apply(split_doc_to_sentences)
print("_sent_list")
# 2) 문장 단위로 행 늘리기(explode)
df_sentences = (
    df[["date", "nid", "_sent_list"]]
      .explode("_sent_list", ignore_index=True)
      .rename(columns={"_sent_list": "sentence"})
)

# 3) 빈 문장 제거 + 정렬(원하신 예시처럼 date, nid 순)
df_sentences["sentence"] = df_sentences["sentence"].fillna("").astype(str).str.strip()
df_sentences = df_sentences[df_sentences["sentence"].ne("")]

#// 정렬: 날짜 오름차순 + nid 오름차순
df_sentences["nid"] = df_sentences["nid"].astype(str).str.strip()
df_sentences = df_sentences.sort_values(["date", "nid"], ascending=[True, True]).reset_index(drop=True)

out_dir = r"C:\Users\User\Desktop\kdt\pjt_bok_1"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "df_sentences.csv")
df_sentences.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Saved to:", out_path)

#전처리 추가
# 불필요 용어 정의
bad_terms = [
    "동양종합금융증권", "우리투자증권", "한국산업은행", "한우대투증권", "삼성증권",
    "한국투자증권", "현대증권", "부국증권", "대우증권", "한양증권", "키움증권",
    "생각대로티제이차유동화", "엘에이치마이홈이차유동화", "색동이제십일차유동화",
    "한화증권", "매도전략", "매수전략", "기하평균", "그림"
]
pattern = "|".join(map(re.escape, bad_terms))

# 불필요 용어가 포함되어있거나 30글자보다 짧은 문장 삭제
df_sentences_cleaned = df_sentences.drop(
    df_sentences[
        (df_sentences["sentence"].fillna("").astype(str).str.len() <= 30) |
        (df_sentences["sentence"].fillna("").astype(str).str.contains(pattern, na=False))
    ].index
).reset_index(drop=True)

# csv로 저장
out_path = os.path.join(out_dir, "df_sentences_cleaned.csv")
df_sentences_cleaned.to_csv(out_path,index=False,encoding="utf-8-sig")

#형태소 분리하기
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

df_sentences_cleaned = df_sentences_cleaned.copy()
df_sentences_cleaned["sent_idx"] = df_sentences_cleaned.groupby(["date", "nid"]).cumcount()

_MECAB = None
def _get_mecab():
    global _MECAB
    if _MECAB is None:
        from ekonlpy import Mecab
        _MECAB = Mecab()
    return _MECAB

def sentence_to_pos_tokens(sent):
    if sent is None or (isinstance(sent, float) and pd.isna(sent)):
        return []
    s = str(sent).strip()
    if not s:
        return []
    mecab = _get_mecab()
    return [f"{w}/{pos}" for (w, pos) in mecab.pos(s)]

sent_list = df_sentences_cleaned["sentence"].tolist()

n_jobs = 6
batch_size = 1500
chunk_size = 50_000

pos_tokens_all = []
for start in range(0, len(sent_list), chunk_size):
    part = sent_list[start:start + chunk_size]

    out = Parallel(n_jobs=n_jobs, backend="loky", batch_size=batch_size)(
        delayed(sentence_to_pos_tokens)(s)
        for s in tqdm(part, total=len(part), desc=f"Tokenizing {start:,}-{start+len(part):,} (n_jobs={n_jobs})")
    )
    pos_tokens_all.extend(out)

df_sentences_cleaned["pos_tokens_sent"] = pos_tokens_all

df_pos_with_nid = (
    df_sentences_cleaned.sort_values(["date", "nid", "sent_idx"])
    .groupby(["date", "nid"], as_index=False)["pos_tokens_sent"]
    .apply(list)
    .rename(columns={"pos_tokens_sent": "pos_tokens"})
)

df_pos_with_nid["pos_tokens"] = df_pos_with_nid["pos_tokens"].apply(repr)
df_pos = df_pos_with_nid[["date", "pos_tokens"]].copy()

out_dir = r"C:\Users\User\Desktop\kdt\pjt_bok_1"
os.makedirs(out_dir, exist_ok=True)

df_pos.to_csv(os.path.join(out_dir, "df_pos_tokens.csv"), index=False, encoding="utf-8-sig")
df_pos_with_nid.to_csv(os.path.join(out_dir, "df_pos_tokens_with_nid.csv"), index=False, encoding="utf-8-sig")

df_pos_2012 = df_pos[df_pos["date"] >= "2012-01-01"].copy()

out_dir = r"C:\Users\User\Desktop\kdt\pjt_bok_1"
os.makedirs(out_dir, exist_ok=True)

df_pos_2012.to_csv(os.path.join(out_dir, "debenture_reports_tokenize.csv"),
                   index=False, encoding="utf-8-sig")