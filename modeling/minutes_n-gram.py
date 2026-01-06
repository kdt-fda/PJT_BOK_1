import json, csv, ast
import pandas as pd
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

        try:
            text = json.loads(text)
        except Exception:
            text = ast.literal_eval(text)

        doc_ngrams = []

        for sent in text:
            sent = str(sent).strip()
            if not sent:
                continue

            tokens = mpck.tokenize(sent)
            ngrams = ngramize(tokens, max_n=5)
            doc_ngrams.append(ngrams)

        results.append({
            "release_date": release_date,
            "tokens": json.dumps(doc_ngrams, ensure_ascii=False),
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

import json, csv
import pandas as pd
from collections import Counter

INPUT_PATH = "minutes_tokenize.csv"
OUTPUT_PATH = "minutes_ngramize.csv"
MIN_FREQ = 15

META_STOP = {
    "있/VV", "되/VV", "보/VV", "하/VV", "이/VV",
    "같/VA", "위하/VV", "따르/VV", "대하/VV", "대해/VV",
    "나타나/VV", "말하/VV", "보이/VV",
    "중/NNG", "수/NNG", "점/NNG", "것/NNG",
    "또/MAG", "아울러/MAG", "이어서/MAG", "함께/MAG", "아직/MAG",
    "등/NNG", "및/NNG"
}

DROP_UNIGRAM = True


def loads(cell):
    if pd.isna(cell) or not str(cell).strip():
        return []
    return json.loads(cell)

def is_contiguous_subseq(short, long):
    s, L = len(short), len(long)
    for i in range(L - s + 1):
        if long[i:i+s] == short:
            return True
    return False

def has_nng(ng):
    return any(tok.endswith("/NNG") for tok in ng.split(";"))

def keep_highest_only(ngrams):
    uniq = {}
    for ng in ngrams:
        ng = str(ng).strip().strip('"').rstrip(";")
        parts = [p for p in ng.split(";") if p]
        if parts:
            uniq[ng] = parts
    items = sorted(uniq.items(), key=lambda x: len(x[1]), reverse=True)
    kept, kept_parts = [], []
    for ng, parts in items:
        if any(is_contiguous_subseq(parts, kp) for kp in kept_parts):
            continue
        kept.append(ng)
        kept_parts.append(parts)
    return kept


df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
docs = df["tokens"].apply(loads)

global_freq = Counter()
for doc in docs:
    for sent in doc:
        for ng in sent:
            ng = str(ng).strip().strip('"').rstrip(";")
            if ng:
                global_freq[ng] += 1

keep_vocab = {ng for ng, c in global_freq.items() if c >= MIN_FREQ}
keep_vocab -= META_STOP

if DROP_UNIGRAM:
    keep_vocab = {ng for ng in keep_vocab if ";" in ng}

keep_vocab = {ng for ng in keep_vocab if has_nng(ng)}

selected_docs = []
for doc in docs:
    new_doc = []
    for sent in doc:
        sent_f = []
        for ng in sent:
            ng2 = str(ng).strip().strip('"').rstrip(";")
            if ng2 in keep_vocab:
                sent_f.append(ng2)
        new_doc.append(keep_highest_only(sent_f))
    selected_docs.append(new_doc)

df_out = pd.DataFrame({
    "release_date": df["release_date"],
    "ngrams": [json.dumps(x, ensure_ascii=False) for x in selected_docs]
})

df_out.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_ALL
)

print("문서 수:", len(df_out))
print("선정 vocab 크기:", len(keep_vocab))
