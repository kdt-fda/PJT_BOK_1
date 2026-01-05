import ast
import json
import pandas as pd
from collections import Counter

INPUT_CSV = "minutes_tokens.csv"
OUTPUT_CSV = "minutes_ngrams.csv"

MAX_N = 5
MIN_FREQ = 15

meta_stop = {
    "이_대해","이어서_동위원","또_일부위원","아울러_동위원","이_관련",
    "답변","이_동위원","과_같","있","다","있_것","있_점","나타내","한편_일부위원",
    "수_있","필요_있",
    "있_의견","있_견해","있_언급","있_답변",
    "것_이","것_보인다고","것_예상","것_판단",
    "보이_있","물_이",
    "대해_관련부서"
}

# tokens 파싱 + flatten
def parse_tokens_cell(x):
    """
    CSV에서 읽은 tokens가
    - 이미 list면 그대로
    - 문자열이면 ast.literal_eval로 list로 변환
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            return ast.literal_eval(x)
        except Exception:
            # 파싱 실패하면 빈 리스트로 처리
            return []
    return []

def flatten_tokens(tok):
    """
    tok가
    - [[...],[...]] (문장별 토큰) 이면 flatten해서 1차원으로
    - [...] (이미 1차원) 이면 그대로
    """
    if not tok:
        return []
    if isinstance(tok, list) and len(tok) > 0 and isinstance(tok[0], list):
        out = []
        for sent_tokens in tok:
            out.extend(sent_tokens)
        return out
    return tok

# n-gram 생성 함수
def keep_longest(tokens, max_n=5):
    """
    논문 규칙에 가까움:
    - 각 위치에서 가능한 최대 n-gram(최장)만 남김
    - i는 최장 길이만큼 점프
    - 최장이 1이면 1-gram이 자연스럽게 포함됨
    """
    L = len(tokens)
    out = []
    i = 0
    while i < L:
        n = min(max_n, L - i)  # 남은 길이 내 최장
        out.append("_".join(tokens[i:i+n]))
        i += n
    return out



df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

# tokens 컬럼 파싱
df["tokens"] = df["tokens"].apply(parse_tokens_cell)
df["tokens_flat"] = df["tokens"].apply(flatten_tokens)

# ngram 생성
df["ngrams_raw"] = df["tokens_flat"].apply(lambda t: keep_longest_plus_bigram(t, max_n=MAX_N))

# 전체 빈도 기반 필터
all_ngrams = df["ngrams_raw"].explode().dropna()
ngram_counts = Counter(all_ngrams)

valid_ngrams = {g for g, c in ngram_counts.items() if c >= MIN_FREQ}
df["ngrams"] = df["ngrams_raw"].apply(lambda gs: [g for g in gs if g in valid_ngrams])

# meta_stop 제거
df["ngrams"] = df["ngrams"].apply(lambda gs: [g for g in gs if g not in meta_stop])

# 저장: release_date + ngrams(리스트만 JSON 문자열로)
out = df[["release_date", "ngrams"]].copy()
out["ngrams"] = out["ngrams"].apply(lambda x: json.dumps(x, ensure_ascii=False))

out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# 체크
final_counts = Counter(df["ngrams"].explode().dropna())
print("saved:", OUTPUT_CSV)
print("최종 ngram vocab size:", len(final_counts))
print("상위 20개:", final_counts.most_common(20))
