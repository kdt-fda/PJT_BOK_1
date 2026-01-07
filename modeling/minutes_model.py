import pandas as pd
import numpy as np
import ast
from collections import Counter
from ekonlpy.sentiment import MPCK
from multiprocessing import Pool
from tqdm import tqdm
import os

os.environ["PYTHONUTF8"] = "1"

master_mpck = MPCK()
h_lexicon_set = set()
d_lexicon_set = set()

def init_worker(h_set=None, d_set=None):
    global h_lexicon_set, d_lexicon_set
    if h_set is not None:
        h_lexicon_set = h_set
    if d_set is not None:
        d_lexicon_set = d_set

def prepare_data(minutes_file, call_rate_file):
    minutes_df = pd.read_csv(minutes_file, encoding="utf-8-sig")
    minutes_df["date"] = pd.to_datetime(minutes_df["release_date"])
    minutes_df["pos_tokens"] = minutes_df["tokens"].apply(ast.literal_eval)

    call_df = pd.read_csv(call_rate_file)
    call_df["TIME"] = pd.to_datetime(call_df["TIME"])

    all_dates = pd.date_range(start="2012-01-01", end="2025-12-31")
    call_df = call_df.set_index("TIME").reindex(all_dates)
    call_df = call_df.fillna(method="ffill").fillna(method="bfill").reset_index()
    call_df.columns = ["TIME", "DATA_VALUE"]

    call_df["label"] = call_df["DATA_VALUE"].diff(30).shift(-30).apply(
        lambda x: 1 if x >= 0.03 else (-1 if x <= -0.03 else 0)
    )

    merged_df = pd.merge(
        minutes_df,
        call_df[["TIME", "label"]],
        left_on="date",
        right_on="TIME",
        how="left",
    )
    return merged_df


def count_features(row):
    all_features = []
    for sent_tokens in row[1]:
        ngrams = master_mpck.ngramize(sent_tokens)
        all_features.extend(sent_tokens + ngrams)
    return row[0], all_features

def build_market_lexicon(df, min_freq=15, threshold=1.3):
    print("\n[Market Approach] 사전 구축 중...")
    train_df = df.dropna(subset=["label"])
    raw_data = train_df[["label", "pos_tokens"]].values.tolist()

    h_counts, d_counts = Counter(), Counter()
    total_h, total_d = 0, 0

    with Pool(processes=2, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(count_features, raw_data, chunksize=50), total=len(raw_data)))

    for label, features in results:
        if label == 1:
            h_counts.update(features)
            total_h += len(features)
        elif label == -1:
            d_counts.update(features)
            total_d += len(features)

    lexicon = []
    for ngram in (set(h_counts.keys()) | set(d_counts.keys())):
        if (h_counts[ngram] + d_counts[ngram]) < min_freq:
            continue
        p_h = h_counts[ngram] / total_h if total_h > 0 else 0
        p_d = d_counts[ngram] / total_d if total_d > 0 else 0
        score = p_h / p_d if p_d > 0 else 0
        if score >= threshold:
            lexicon.append({"ngram": ngram, "polarity": "hawkish", "score": score})
        elif score <= (1 / threshold):
            lexicon.append({"ngram": ngram, "polarity": "dovish", "score": score})

    return pd.DataFrame(lexicon)

def tone_worker(pos_tokens):
    global h_lexicon_set, d_lexicon_set
    sent_labels = []
    for sent in pos_tokens:
        features = sent + master_mpck.ngramize(sent)
        h = sum(1 for f in features if f in h_lexicon_set)
        d = sum(1 for f in features if f in d_lexicon_set)
        if h > d:
            sent_labels.append(1)
        elif d > h:
            sent_labels.append(-1)
    nh, nd = sent_labels.count(1), sent_labels.count(-1)
    return (nh - nd) / (nh + nd) if (nh + nd) > 0 else 0

def run_multiprocess_tone(df, lexicon_df, method_name):
    print(f"\n[{method_name}] Tone Score 계산 중...")
    h_set = set(lexicon_df[lexicon_df["polarity"] == "hawkish"]["ngram"])
    d_set = set(lexicon_df[lexicon_df["polarity"] == "dovish"]["ngram"])
    with Pool(processes=2, initializer=init_worker, initargs=(h_set, d_set)) as pool:
        results = list(tqdm(pool.imap(tone_worker, df["pos_tokens"], chunksize=50), total=len(df)))
    return results

if __name__ == "__main__":
    all_data = prepare_data("minutes_tokenize.csv", "call_rate.csv")

    train_data = all_data[all_data["date"] < "2025-12-01"].copy()

    m_lex = build_market_lexicon(train_data)
    m_lex.to_csv("minutes_market_approach_dict.csv", index=False, encoding="utf-8-sig")

    all_data["tone_market"] = run_multiprocess_tone(all_data, m_lex, "Market")

    all_data[["release_date", "tone_market", "label"]].to_csv(
        "minutes_tone_score.csv", index=False, encoding="utf-8-sig"
    )

    print("\n완료되었습니다.")