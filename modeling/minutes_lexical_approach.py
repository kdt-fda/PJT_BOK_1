import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

INPUT_PATH = "minutes_ngramize.csv"
COL_NGRAMS = "ngrams"

def loads_doc(cell):
    if pd.isna(cell) or not str(cell).strip():
        return []
    x = json.loads(cell)
    return x if isinstance(x, list) else []

def clean_ng(ng: str) -> str:
    return str(ng).strip().strip('"').rstrip(";")

df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
docs = df[COL_NGRAMS].apply(loads_doc)



HAWKISH_SEEDS = [
    "높/VA","팽창/NNG","인상/NNG","매파/NNG","성장/NNG","투기/NNG;억제/NNG","상승/NNG",
    "인플레이션/NNG;압력/NNG","증가/NNG","위험/NNG;선호/NNG","상회/NNG","물가/NNG;상승/NNG",
    "과열/NNG","금리/NNG;상승/NNG","확장/NNG","상방/NNG;압력/NNG","긴축/NNG","변동성/NNG;감소/NNG",
    "흑자/NNG","채권/NNG;가격/NNG;하락/NNG","견조/NNG","요금/NNG;인상/NNG","낙관/NNG",
    "부동산/NNG;가격/NNG;상승/NNG","상향/NNG",
]
DOVISH_SEEDS = [
    "낮/VA","축소/NNG","인하/NNG","비둘기/NNG","둔화/NNG","악화/NNG","하락/NNG","회복/NNG;못하/VX",
    "감소/NNG","위험/NNG;회피/NNG","하회/NNG","물가/NNG;하락/NNG","위축/NNG","금리/NNG;하락/NNG",
    "침체/NNG","하방/NNG;압력/NNG","완화/NNG","변동성/NNG;확대/NNG","적자/NNG","채권/NNG;가격/NNG;상승/NNG",
    "부진/NNG","요금/NNG;인하/NNG","비관/NNG","부동산/NNG;가격/NNG;하락/NNG","하향/NNG",
]



import numpy as np
from gensim.models import Word2Vec

w2v = Word2Vec.load("ngram_w2v.model")

model_vocab = list(w2v.wv.index2word)
print("vocab size:", len(model_vocab))

X = np.vstack([w2v.wv[w] for w in model_vocab]).astype(np.float32)
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # cosine용 정규화

word2i = {w:i for i,w in enumerate(model_vocab)}

hawk_seed = [s for s in HAWKISH_SEEDS if s in word2i]
dove_seed = [s for s in DOVISH_SEEDS if s in word2i]
print("hawk seeds found:", len(hawk_seed), "/", len(HAWKISH_SEEDS))
print("dove seeds found:", len(dove_seed), "/", len(DOVISH_SEEDS))

#knn 그래프
from sklearn.neighbors import NearestNeighbors

TOPK = 25  # 보통 25~50
nn = NearestNeighbors(n_neighbors=TOPK+1, metric="cosine")
nn.fit(X)

dist, idx = nn.kneighbors(X, return_distance=True)
idx = idx[:, 1:]      # 자기 자신 제거
sim = 1.0 - dist[:, 1:]
sim = np.clip(sim, 0.0, None)

row_sum = sim.sum(axis=1, keepdims=True) + 1e-12
P = sim / row_sum     # row-stochastic transition

#sentprop
def ppr_scores(seed_words, alpha=0.85, max_iter=200, tol=1e-8):
    N = len(model_vocab)
    seeds = [word2i[w] for w in seed_words if w in word2i]
    if len(seeds) == 0:
        return np.zeros(N, dtype=np.float32)

    s = np.zeros(N, dtype=np.float32)
    s[seeds] = 1.0 / len(seeds)

    r = s.copy()
    for _ in range(max_iter):
        r_new = (1 - alpha) * s
        contrib = alpha * r[:, None] * P  # (N, TOPK)

        r_new_acc = r_new.copy()
        for i in range(N):
            r_new_acc[idx[i]] += contrib[i].astype(np.float32)

        r_new_acc /= (r_new_acc.sum() + 1e-12)

        if np.max(np.abs(r_new_acc - r)) < tol:
            r = r_new_acc
            break
        r = r_new_acc

    return r

hawk_prob = ppr_scores(hawk_seed, alpha=0.85)
dove_prob = ppr_scores(dove_seed, alpha=0.85)

import pandas as pd

INTENSITY = 1.3
eps = 1e-12

score = (hawk_prob + eps) / (dove_prob + eps)

labels = np.full(len(model_vocab), "gray", dtype=object)
labels[score > INTENSITY] = "hawkish"
labels[score < 1.0/INTENSITY] = "dovish"

lex = pd.DataFrame({
    "lexicon": model_vocab,
    "polarity_score": score,
    "label": labels
})

lex_final = lex[lex["label"] != "gray"].copy()
print("hawkish:", (lex_final["label"]=="hawkish").sum())
print("dovish :", (lex_final["label"]=="dovish").sum())
print("total  :", len(lex_final))

lex_final.to_csv("lexicon_lexical_minutes.csv", index=False, encoding="utf-8-sig")


import json
import numpy as np
import pandas as pd

MINUTES_PATH = "minutes_ngramize.csv"
LEXICON_PATH = "lexicon_lexical_minutes.csv"   # lexicon, polarity_score, label
OUT_PATH = "lexical_tone_index_minutes.csv"


minutes = pd.read_csv(MINUTES_PATH, encoding="utf-8-sig")[["release_date", "ngrams"]].copy()
minutes["release_date"] = pd.to_datetime(minutes["release_date"], errors="coerce")

def loads_tokens(x):
    if pd.isna(x) or not str(x).strip():
        return []
    obj = json.loads(x)
    toks = []
    for sent in obj:
        if isinstance(sent, list):
            for t in sent:
                t = str(t).strip().strip('"').rstrip(";")
                if t:
                    toks.append(t)
    return toks

minutes["tokens"] = minutes["ngrams"].apply(loads_tokens)


lex = pd.read_csv(LEXICON_PATH, encoding="utf-8-sig")[["lexicon", "polarity_score", "label"]].copy()
lex["polarity_score"] = pd.to_numeric(lex["polarity_score"], errors="coerce")
lex["label"] = lex["label"].astype(str).str.lower().str.strip()

# hawkish/dovish만 사용 (neutral은 제외)
lex = lex[lex["label"].isin(["hawkish", "dovish"])].dropna(subset=["polarity_score"])

# 부호 적용
lex["signed_score"] = np.where(lex["label"] == "hawkish", lex["polarity_score"], -lex["polarity_score"])

score_map = dict(zip(lex["lexicon"], lex["signed_score"]))

# 문서별 tone index 계산
def doc_tone_avg(tokens, score_map):
    s = 0.0
    m = 0
    for t in tokens:
        sc = score_map.get(t)
        if sc is None:
            continue
        s += float(sc)
        m += 1
    return (s / m) if m > 0 else np.nan, m

def doc_tone_posneg(tokens, score_map):
    P = 0
    N = 0
    for t in tokens:
        sc = score_map.get(t)
        if sc is None:
            continue
        if sc > 0:
            P += 1
        elif sc < 0:
            N += 1
    return (P - N) / (P + N) if (P + N) > 0 else np.nan, (P + N)

tone_avg, tone_pn, match_cnt, match_cnt_pn = [], [], [], []

for toks in minutes["tokens"]:
    a, m = doc_tone_avg(toks, score_map)
    b, m2 = doc_tone_posneg(toks, score_map)
    tone_avg.append(a)
    tone_pn.append(b)
    match_cnt.append(m)
    match_cnt_pn.append(m2)

out = pd.DataFrame({
    "release_date": minutes["release_date"],
    "lex_tone_avg": tone_avg,
    "lex_tone_posneg": tone_pn,
    "lex_match_count": match_cnt,
    "lex_match_count_posneg": match_cnt_pn,
    "doc_token_count": minutes["tokens"].apply(len),
}).sort_values("release_date")

out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print("Saved:", OUT_PATH)
print(out.tail(5))
