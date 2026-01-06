import json
import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


class NaiveBayesMarketMinutes:
    def __init__(self, k=0.5, horizon_days=30):
        self.k = k
        self.horizon_days = horizon_days
        self.word_probs = []  # (w, p_w_hawk, p_w_dove)
        self.scores = {}      # log(p_w_hawk/p_w_dove)

        self.minutes_df = None
        self.call_series = None

    # ---------- 데이터 로드 ----------
    def load_minutes(self, path="minutes_ngramize.csv"):
        df = pd.read_csv(path, encoding="utf-8-sig")[["release_date", "ngrams"]].copy()
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

        def loads_cell(x):
            if pd.isna(x) or not str(x).strip():
                return []
            obj = json.loads(x)  # [[...],[...],...]
            toks = []
            for sent in obj:
                if isinstance(sent, list):
                    for t in sent:
                        t = str(t).strip().strip('"').rstrip(";")
                        if t:
                            toks.append(t)
            return toks

        df["tokens"] = df["ngrams"].apply(loads_cell)
        self.minutes_df = df.dropna(subset=["release_date"]).copy()

    def load_call_rate(self, path="call_rate.csv"):
        cr = pd.read_csv(path, encoding="utf-8-sig")[["TIME", "DATA_VALUE"]].copy()
        cr["TIME"] = pd.to_datetime(cr["TIME"], errors="coerce")
        cr["DATA_VALUE"] = pd.to_numeric(cr["DATA_VALUE"], errors="coerce")
        cr = cr.dropna().sort_values("TIME").set_index("TIME")

        cr.loc[pd.Timestamp("2025-12-31"), "DATA_VALUE"] = 2.625

        self.call_series = cr["DATA_VALUE"]

    # market label 생성 
    def make_training_set(self):
        """
        return: np.array of (release_date_str, label)
        label: 1 hawkish, 0 dovish
        """
        rows = []
        for d in self.minutes_df["release_date"]:
            if d.year == 2025 and d.month == 12:
                continue

            t0 = d
            t1 = d + pd.Timedelta(days=self.horizon_days)

            # 주말/공휴일: 직전값(asof)
            try:
                r0 = self.call_series.loc[:t0].iloc[-1]
                r1 = self.call_series.loc[:t1].iloc[-1]
            except Exception:
                continue

            if r1 > r0:
                y = 1
            elif r1 < r0:
                y = 0
            else:
                continue  # 변화 0이면 제외(논문처럼 +/-만 사용)

            rows.append((t0.strftime("%Y-%m-%d"), y))

        return np.array(rows, dtype=object)

    # class별 카운트 
    def count_words(self, training_set):
        counts = defaultdict(lambda: [0, 0])  # [hawk, dove]

        # 빠른 매칭을 위해 dict로
        date2tokens = dict(zip(
            self.minutes_df["release_date"].dt.strftime("%Y-%m-%d"),
            self.minutes_df["tokens"]
        ))

        for dataDate, label in training_set:
            toks = date2tokens.get(str(dataDate))
            if not toks:
                continue

            for w in toks:
                counts[w][0 if int(label) == 1 else 1] += 1

        return counts

    # 조건부확률(naive bayes 형태)
    def word_probabilities(self, counts, k):
        hawk_total = sum(v[0] for v in counts.values())
        dove_total = sum(v[1] for v in counts.values())
        V = len(counts)

        probs = []
        for w, (hawk_c, dove_c) in counts.items():
            p_w_hawk = (hawk_c + k) / (hawk_total + k * V)
            p_w_dove = (dove_c + k) / (dove_total + k * V)
            probs.append((w, p_w_hawk, p_w_dove))
        return probs

    def train(self, minutes_path="minutes_ngramize.csv", call_rate_path="call_rate.csv", out_dir="."):
        import os
        os.makedirs(out_dir, exist_ok=True)

        self.load_minutes(minutes_path)
        self.load_call_rate(call_rate_path)

        training_set = self.make_training_set()
        counts = self.count_words(training_set)

        self.word_probs = self.word_probabilities(counts, self.k)

        # polarity score (연속값) 저장: 논문 수식과 동일한 의미
        self.scores = {w: math.log(p1 / p0) for w, p1, p0 in self.word_probs}

        # 컬럼/파일명 통일
        lexicon_df = pd.DataFrame(
            [{"lexicon": w, "polarity_score": s} for w, s in self.scores.items()]
        ).sort_values("polarity_score", ascending=False)

        lexicon_df.to_csv(os.path.join(out_dir, "lexicon_market_minutes.csv"),
                          index=False, encoding="utf-8-sig")

        return lexicon_df



if __name__ == "__main__":
    model = NaiveBayesMarketMinutes(k=0.5, horizon_days=30)
    score_df = model.train(
        minutes_path="minutes_ngramize.csv",
        call_rate_path="call_rate.csv",
        out_dir="."
    )


import json
import numpy as np
import pandas as pd


MINUTES_PATH = "minutes_ngramize.csv"
SCORES_PATH = "lexicon_market_minutes.csv"
OUT_PATH = "market_tone_index_minutes.csv"

minutes = pd.read_csv(MINUTES_PATH, encoding="utf-8-sig")[["release_date", "ngrams"]].copy()
minutes["release_date"] = pd.to_datetime(minutes["release_date"], errors="coerce")

scores_df = pd.read_csv(SCORES_PATH, encoding="utf-8-sig")[["lexicon", "polarity_score"]].copy()
score_map = dict(zip(scores_df["lexicon"], scores_df["polarity_score"]))

def loads_tokens(x):
    if pd.isna(x) or not str(x).strip():
        return []
    obj = json.loads(x)  # [[...],[...],...]
    toks = []
    for sent in obj:
        if isinstance(sent, list):
            for t in sent:
                t = str(t).strip().strip('"').rstrip(";")
                if t:
                    toks.append(t)
    return toks

minutes["tokens"] = minutes["ngrams"].apply(loads_tokens)


# 문서별 tone index 계산

def doc_tone_avglogodds(tokens, score_map):
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

tone_avg = []
tone_pn = []
match_cnt = []
match_cnt_pn = []

for toks in minutes["tokens"]:
    a, m = doc_tone_avglogodds(toks, score_map)
    b, m2 = doc_tone_posneg(toks, score_map)
    tone_avg.append(a)
    tone_pn.append(b)
    match_cnt.append(m)
    match_cnt_pn.append(m2)

out = pd.DataFrame({
    "release_date": minutes["release_date"],
    "tone_avglogodds": tone_avg,     # 추천 메인 지수
    "tone_posneg": tone_pn,          # (P-N)/(P+N) 보조 지수
    "match_count": match_cnt,        # score_map에 매칭된 토큰 수
    "match_count_posneg": match_cnt_pn,
    "doc_token_count": minutes["tokens"].apply(len),
})

out = out.sort_values("release_date")
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(out.tail(5))
