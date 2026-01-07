# 라이브러리
import os
import ast
import math
import time
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix

# 초기 변수 설정
# 파일 경로
DEBENTURE_PATH = "debenture_page_tokenize.csv"   # date, pos_tokens
CALLCHANGE_PATH = "callrate_1m_change.csv"          # date, call_1m_change

# 논문 규칙 파라미터 
N_MAX = 5
MIN_FREQ = 15
STOP_POS = {"VA","NNP","JKO","EP","ETM","VV","VCP","NNB","SY","NNBC",\
            "MM","SF","EF","XSV","ETN","XSA","JX","JKB","JKS","EC","VX",} #불용어태그

THRESHOLD_PP = 0.03      # ±3dp 제외. %p 기준이면 0.03
N_BAGS = 30              # bagging 30회
INTENSITY = 1.3          # gray 제외 (>=1.3 hawk, <=1/1.3 dove)

RANDOM_SEED = 42
TEST_SIZE = 0.1
ALPHA = 1.0

# 채권 보고서 토큰 가져오기
def load_debenture_tokens(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"date", "pos_tokens"}.issubset(df.columns):
        raise ValueError("debenture_page_tokenize.csv must have columns: date, pos_tokens")
    df["date"] = pd.to_datetime(df["date"])
    df["pos_tokens"] = df["pos_tokens"].apply(ast.literal_eval)  # list[list[str]]
    return df

# 콜금리 변화량 가져오기(변화량으로 계산해둠)
def load_callrate_1m_change(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"date", "call_1m_change"}.issubset(df.columns):
        raise ValueError("callrate_1m_change.csv must have columns: date, call_1m_change")
    df["date"] = pd.to_datetime(df["date"])
    df["call_1m_change"] = pd.to_numeric(df["call_1m_change"], errors="coerce")
    df = df.dropna(subset=["date", "call_1m_change"]).sort_values("date")
    return df

# 데이터 프레임에 할당
df_docs = load_debenture_tokens(DEBENTURE_PATH)
call_df = load_callrate_1m_change(CALLCHANGE_PATH)

### 전처리(특정 POS만) + "겹치면 가장 긴 n-gram만" 함수 정의
def split_wp(token: str) -> Tuple[str, str]:
    if "/" not in token:
        return token, ""
    w, p = token.rsplit("/", 1)
    return w, p

def normalize_token(token: str, stop_pos: Optional[set]):
    if "/" not in token:
        return None

    w, p = token.rsplit("/", 1)
    if not w or not p:
        return None

    # stop POS에 포함되면 제거
    if stop_pos is not None and p in stop_pos:
        return None

    return f"{w}/{p}"

def normalize_sentences(pos_tokens, stop_pos):
    out = []
    for sent in pos_tokens:
        norm = []
        for t in sent:
            nt = normalize_token(t, stop_pos)
            if nt is not None:
                norm.append(nt)
        out.append(norm)
    return out


def sentence_highest_ngrams(tokens: List[str], n_max: int = 5) -> List[Tuple[str, ...]]:
    L = len(tokens)
    covered = [False] * L
    selected: List[Tuple[str, ...]] = []

    for n in range(n_max, 0, -1):
        if L < n:
            continue
        for i in range(0, L - n + 1):
            span = range(i, i + n)
            if any(covered[j] for j in span):
                continue
            ng = tuple(tokens[i:i+n])
            selected.append(ng)
            for j in span:
                covered[j] = True
    return selected



### STEP 2) (PASS 1) 전체 코퍼스 n-gram 빈도 집계 → MIN_FREQ=15로 vocab 만들기 → 실행
def pass1_build_global_freq(
    df_docs: pd.DataFrame,
    stop_pos: Optional[set],
    n_max: int = 5,
    log_every_docs: int = 500,
):
    global_freq = Counter()

    for i, pos_tokens in enumerate(df_docs["pos_tokens"]):
        sents = normalize_sentences(pos_tokens, stop_pos)
        for sent in sents:
            ngs = sentence_highest_ngrams(sent, n_max=n_max)
            global_freq.update(ngs)

    return global_freq

def build_vocab(global_freq: Counter, min_freq: int = 15) -> Dict[Tuple[str, ...], int]:
    items = [ng for ng, c in global_freq.items() if c >= min_freq]
    items.sort()
    return {ng: j for j, ng in enumerate(items)}


global_freq = pass1_build_global_freq(df_docs, STOP_POS, n_max=N_MAX, log_every_docs=500)
vocab = build_vocab(global_freq, min_freq=MIN_FREQ)

### 이 단계는 "문서별 ngram 저장"을 하지 않고, 빈도만 집계(메모리 폭발 방지)
# 날짜 겹치는 구간에서만 콜금리 변화량을 봄
doc_dates = set(df_docs["date"])
call_sub = call_df[call_df["date"].isin(doc_dates)].copy()

x = call_sub["call_1m_change"].astype(float).to_numpy()

print("abs(change) quantiles:", np.quantile(np.abs(x), [0.5, 0.9, 0.99]))
print("abs(change) max:", np.max(np.abs(x)))

for th in [0.03, 0.02, 0.01, 0.005, 0.003, 0.001]:
    keep = (np.abs(x) > th).mean()
    print(f"threshold {th}: keep ratio {keep:.3f} | kept days ~ {int(keep*len(x))}")

# vocab에 있는 n-gram만 feature로 유지
# 라벨링: call_1m_change > +0.03 hawkish(1), < -0.03 dovish(0), 그 외 제외(-1)

def build_call_lookup(call_df: pd.DataFrame) -> Dict[pd.Timestamp, float]:
    return dict(zip(call_df["date"], call_df["call_1m_change"]))

def pass2_build_sentence_matrix(
    df_docs: pd.DataFrame,
    call_lookup: Dict[pd.Timestamp, float],
    vocab: Dict[Tuple[str, ...], int],
    stop_pos: set,
    n_max: int = 5,
    threshold_pp: float = 0.03,
    log_every_docs: int = 500,
):

    row_idx = []
    col_idx = []
    data = []

    y = []
    meta = []  # (doc_idx, sent_idx, date)

    n_docs = len(df_docs)
    n_feat = len(vocab)

    unmatched_dates = 0
    kept_sentences = 0
    total_sentences_with_feats = 0

    for doc_i, (date, pos_tokens) in enumerate(zip(df_docs["date"], df_docs["pos_tokens"])):
        if date not in call_lookup:
            unmatched_dates += 1
            continue

        change = call_lookup[date]
        if abs(change) <= threshold_pp:
            continue

        label = 1 if change > threshold_pp else 0

        sents = normalize_sentences(pos_tokens, stop_pos)
        for sent_j, sent in enumerate(sents):
            ngs = sentence_highest_ngrams(sent, n_max=n_max)
            feats = [ng for ng in ngs if ng in vocab]
            if not feats:
                continue

            total_sentences_with_feats += 1

            # sparse row construction
            r = len(y)
            counts = Counter(feats)
            for ng, cnt in counts.items():
                row_idx.append(r)
                col_idx.append(vocab[ng])
                data.append(int(cnt))

            y.append(label)
            meta.append((doc_i, sent_j, date))
            kept_sentences += 1

    if kept_sentences == 0:
        raise RuntimeError("PASS2 결과 문장이 0개입니다. 날짜 매칭/threshold_pp/vocab을 점검하세요.")

    X = csr_matrix((data, (row_idx, col_idx)), shape=(kept_sentences, n_feat), dtype=np.int32)
    y = np.array(y, dtype=np.int8)
    meta = pd.DataFrame(meta, columns=["doc_idx", "sent_idx", "date"])

    print(f"[PASS2 DONE] X shape={X.shape} | nnz={X.nnz} | y counts={np.bincount(y)}")
    print(f"unmatched_dates(docs): {unmatched_dates}  (콜금리 파일에 없는 날짜)")

    return X, y, meta

call_lookup = build_call_lookup(call_df)

X, y, meta = pass2_build_sentence_matrix(
    df_docs=df_docs,
    call_lookup=call_lookup,
    vocab=vocab,
    stop_pos=STOP_POS,
    n_max=N_MAX,
    threshold_pp=0.03
)

### STEP 4) NBC polarity score (식(1)) + bagging 30회 평균 → 실행
def train_nb_polarity_bagging(
    X: csr_matrix,
    y: np.ndarray,
    n_bags: int = 30,
    test_size: float = 0.1,
    seed: int = 42,
    alpha: float = 1.0,
):
    rng = np.random.RandomState(seed)
    scores = []

    for b in range(n_bags):
        rs = int(rng.randint(0, 10**9))
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)

        nb = MultinomialNB(alpha=alpha)
        nb.fit(X_tr, y_tr)

        classes = list(nb.classes_)
        if 0 not in classes or 1 not in classes:
            raise RuntimeError("학습 분할에서 한쪽 클래스가 사라졌습니다. 데이터/threshold를 점검하세요.")

        idx_dove = classes.index(0)
        idx_hawk = classes.index(1)

        # p(feature|class)
        p_feat_hawk = np.exp(nb.feature_log_prob_[idx_hawk])
        p_feat_dove = np.exp(nb.feature_log_prob_[idx_dove])

        # p(class)
        p_hawk = float(np.exp(nb.class_log_prior_[idx_hawk]))
        p_dove = float(np.exp(nb.class_log_prior_[idx_dove]))

        # polarity score = p(f|hawk)*p(hawk) / (p(f|dove)*p(dove))
        pol = (p_feat_hawk * p_hawk) / (p_feat_dove * p_dove)
        scores.append(pol)

        if (b + 1) % 5 == 0:
            print(f"[NB bagging] {b+1}/{n_bags}")

    polarity_score = np.mean(np.vstack(scores), axis=0)
    return polarity_score


polarity_score = train_nb_polarity_bagging(
    X, y, n_bags=N_BAGS, test_size=TEST_SIZE, seed=RANDOM_SEED, alpha=ALPHA
)

### STEP 5) hawkish/dovish 사전 구축(강도 1.3 gray 제외) + 저장 → 실행

def build_dict(
    vocab: Dict[Tuple[str, ...], int],
    polarity_score: np.ndarray,
    intensity: float = 1.3,
):
    hawk = {}
    dove = {}
    for ng, j in vocab.items():
        s = float(polarity_score[j])
        if s >= intensity:
            hawk[ng] = s
        elif s <= 1.0 / intensity:
            dove[ng] = s
    return hawk, dove

hawk_lex, dove_lex = build_dict(vocab, polarity_score, intensity=INTENSITY)

print("hawk_lex size:", len(hawk_lex))
print("dove_lex size:", len(dove_lex))

def save_dict_csv(hawk_lex, dove_lex, out_path="page_dict_market.csv"):
    rows = []
    for ng, s in hawk_lex.items():
        rows.append({"ngram": ";".join(ng), "class": "hawkish", "polarity_score": s})
    for ng, s in dove_lex.items():
        rows.append({"ngram": ";".join(ng), "class": "dovish", "polarity_score": s})
    pd.DataFrame(rows).sort_values(["class", "polarity_score"], ascending=[True, False]) \
        .to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

out_lex = save_dict_csv(hawk_lex, dove_lex, out_path="page_dict_market.csv")

### STEP 6) 문서 tone 산출(스트리밍 3rd pass) + 저장 → 실행
def score_documents_streaming(
    df_docs: pd.DataFrame,
    hawk_lex: Dict[Tuple[str, ...], float],
    dove_lex: Dict[Tuple[str, ...], float],
    allowed_pos: set,
    n_max: int = 5,
    log_every_docs: int = 500,
):

    out_rows = []

    hawk_set = set(hawk_lex.keys())
    dove_set = set(dove_lex.keys())

    for i, (date, pos_tokens) in enumerate(zip(df_docs["date"], df_docs["pos_tokens"])):
        sents = normalize_sentences(pos_tokens, allowed_pos)

        sent_scores = []
        for sent in sents:
            ngs = sentence_highest_ngrams(sent, n_max=n_max)
            hc = sum(1 for ng in ngs if ng in hawk_set)
            dc = sum(1 for ng in ngs if ng in dove_set)
            denom = hc + dc
            sent_scores.append(0.0 if denom == 0 else (hc - dc) / denom)

        tone = float(np.mean(sent_scores)) if sent_scores else 0.0
        out_rows.append({"doc_idx": i, "date": date, "tone": tone})


    doc_tone = pd.DataFrame(out_rows)
    
    return doc_tone


doc_tone = score_documents_streaming(df_docs, hawk_lex, dove_lex, STOP_POS, n_max=N_MAX, log_every_docs=500)
doc_tone.to_csv("debenture_page_doc_tone.csv", index=False, encoding="utf-8-sig")
print(doc_tone.head())
print("Saved: debenture_page_doc_tone.csv")

### 날짜별 톤 점수 만들기
# - normalize_sentences(pos_tokens, allowed_pos)
# - sentence_highest_ngrams(tokens, n_max)

def compute_sentence_tones_for_doc(
    pos_tokens: List[List[str]],
    hawk_set: set,
    dove_set: set,
    stop_pos: set,
    n_max: int = 5
) -> List[float]:
    sents = normalize_sentences(pos_tokens, stop_pos)
    sent_tones = []
    for sent in sents:
        ngs = sentence_highest_ngrams(sent, n_max=n_max)
        hc = sum(1 for ng in ngs if ng in hawk_set)
        dc = sum(1 for ng in ngs if ng in dove_set)
        denom = hc + dc
        sent_tones.append(0.0 if denom == 0 else (hc - dc) / denom)
    return sent_tones


def build_doc_sentence_tone_tables(
    df_docs: pd.DataFrame,
    hawk_lex: Dict[Tuple[str, ...], float],
    dove_lex: Dict[Tuple[str, ...], float],
    stop_pos: set,
    n_max: int = 5,
    log_every_docs: int = 500
):
    hawk_set = set(hawk_lex.keys())
    dove_set = set(dove_lex.keys())

    sent_rows = []
    doc_rows = []

    for doc_idx, (date, pos_tokens) in enumerate(zip(df_docs["date"], df_docs["pos_tokens"])):
        sent_tones = compute_sentence_tones_for_doc(
            pos_tokens, hawk_set, dove_set, stop_pos, n_max=n_max
        )

        # sentence table
        for sent_idx, st in enumerate(sent_tones):
            sent_rows.append({"doc_idx": doc_idx, "sent_idx": sent_idx, "date": date, "sent_tone": st})

        # document tone = mean(sentence tones)
        doc_tone = float(np.mean(sent_tones)) if sent_tones else 0.0
        doc_rows.append({"doc_idx": doc_idx, "date": date, "doc_tone": doc_tone, "n_sents": len(sent_tones)})

        if (doc_idx + 1) % log_every_docs == 0:
            print(f"[TONE] processed docs {doc_idx+1}/{len(df_docs)}")

    df_sent_tone = pd.DataFrame(sent_rows)
    df_doc_tone = pd.DataFrame(doc_rows)

    # date tone = 평균(문서톤) + 문서수 정보 같이 남김
    df_date_tone = (
        df_doc_tone
        .groupby("date", as_index=False)
        .agg(
            date_tone=("doc_tone", "mean"),
            n_docs=("doc_idx", "count"),
            avg_sents_per_doc=("n_sents", "mean")
        )
        .sort_values("date")
    )
    return df_sent_tone, df_doc_tone, df_date_tone


# 실행
df_sent_tone, df_doc_tone, df_date_tone = build_doc_sentence_tone_tables(
    df_docs=df_docs,
    hawk_lex=hawk_lex,
    dove_lex=dove_lex,
    stop_pos=STOP_POS,
    n_max=N_MAX,
    log_every_docs=500
)


df_sent_tone.to_csv("debenture_page_sentence_tone.csv", index=False, encoding="utf-8-sig")
df_doc_tone.to_csv("debenture_page_doc_tone_recalc.csv", index=False, encoding="utf-8-sig")
df_date_tone.to_csv("debenture_page_date_tone.csv", index=False, encoding="utf-8-sig")


