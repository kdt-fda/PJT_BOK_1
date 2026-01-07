import pandas as pd
import numpy as np
import os
import multiprocessing
from collections import Counter
from tqdm.notebook import tqdm
from ekonlpy.sentiment import MPCK
from multiprocessing import Pool
from gensim.models import Word2Vec

# 전역 변수 설정
master_mpck = MPCK()
h_lex_set = set()
d_lex_set = set()

def init_worker(h=None, d=None):
    global master_mpck, h_lex_set, d_lex_set
    master_mpck = MPCK()
    if h is not None: h_lex_set = h
    if d is not None: d_lex_set = d

# 뉴스 데이터 통합
def prepare_news(file_list):
    print('1: 뉴스 데이터 통합 중...')
    df_list = []
    for f in tqdm(file_list, desc="파일 로드"):
        temp_df = pd.read_csv(f)
        df_list.append(temp_df)
    df_total = pd.concat(df_list, ignore_index=True)
    df_total['date'] = pd.to_datetime(df_total['date'])
    return df_total.sort_values('date')

# 2. 콜금리 라벨링 (미래 변동 t+30 기준)
def prepare_call_rate_future(file_path, start_date, end_date):
    print(f'2: 콜금리 데이터 로드 및 미래 변동 라벨링 중...')
    df_rate = pd.read_csv(file_path)
    df_rate = df_rate.rename(columns={'TIME': 'date', 'DATA_VALUE': 'rate'})
    df_rate['date'] = pd.to_datetime(df_rate['date'])
    
    all_dates = pd.date_range(start=start_date, end=end_date)
    df_rate = df_rate.drop_duplicates('date').set_index('date').reindex(all_dates)
    df_rate['rate'] = df_rate.ffill().bfill()
    df_rate = df_rate.reset_index().rename(columns={'index': 'date'})
    
    # 미래 30일 변동 계산
    df_rate['rate_diff'] = df_rate['rate'].diff(30).shift(-30)
    df_rate['label'] = df_rate['rate_diff'].apply(
        lambda x: 1 if x >= 0.03 else (-1 if x <= -0.03 else 0)
    )
    return df_rate

# 3. [시장 접근법] 사전 구축
def process_ngram_task(row):
    label, tokens_str = row
    if pd.isna(tokens_str): return None
    tokens = str(tokens_str).split(',')
    try:
        features = tokens + master_mpck.ngramize(tokens)
        return label, features
    except: return None

def build_market_lexicon(df, min_freq=15, intensity_threshold=1.3):
    train_df = df[df['label'] != 0].copy()
    raw_data = train_df[['label', 'tokens']].values.tolist()
    h_counts, d_counts = Counter(), Counter()
    total_h, total_d = 0, 0
    
    print(f'[Market Approach] 사전 학습 중 (학습 데이터: {len(train_df)}건)')
    num_procs = multiprocessing.cpu_count()
    with Pool(processes=num_procs, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_ngram_task, raw_data, chunksize=500), total=len(raw_data), desc="N-gramizing"))
        
    for res in results:
        if res is None: continue
        label, features = res
        if label == 1: h_counts.update(features); total_h += len(features)
        elif label == -1: d_counts.update(features); total_d += len(features)

    lexicon = []
    all_features = set(h_counts.keys()) | set(d_counts.keys())
    epsilon = 1e-7 
    for ngram in tqdm(all_features, desc="Calculating Polarity"):
        f_h, f_d = h_counts[ngram], d_counts[ngram]
        if (f_h + f_d) < min_freq: continue
        p_h, p_d = (f_h + epsilon) / (total_h + epsilon), (f_d + epsilon) / (total_d + epsilon)
        score = p_h / p_d
        if score >= intensity_threshold: lexicon.append({'ngram': ngram, 'polarity': 'hawkish', 'score': score})
        elif score <= (1 / intensity_threshold): lexicon.append({'ngram': ngram, 'polarity': 'dovish', 'score': score})
    return pd.DataFrame(lexicon)

# 4. [어휘적 접근법] 사전 구축
def build_lexical_lexicon(df, intensity=1.1):
    class SentenceIterator:
        def __init__(self, token_series): self.token_series = token_series
        def __iter__(self):
            for t in self.token_series:
                if pd.notna(t): yield str(t).split(',')

    print(f'[Lexical Approach] Word2Vec 학습 중 (학습 데이터 기준)...')
    model = Word2Vec(sentences=SentenceIterator(df['tokens']), vector_size=300, window=5, min_count=15, workers=multiprocessing.cpu_count(), sg=0)
    
    seeds_h = ['인상/NNG', '상승/NNG', '확대/NNG', '긴축/NNG', '낙관/NNG', '강화/NNG', '과열/NNG', '호조/NNG']
    seeds_d = ['인하/NNG', '하락/NNG', '축소/NNG', '완화/NNG', '비관/NNG', '약화/NNG', '둔화/NNG', '침체/NNG']
    vocab = model.wv.index_to_key
    seeds_h = [s for s in seeds_h if s in vocab]; seeds_d = [s for s in seeds_d if s in vocab]
    
    avg_vec_h, avg_vec_d = np.mean([model.wv[s] for s in seeds_h], axis=0), np.mean([model.wv[s] for s in seeds_d], axis=0)
    model.wv.fill_norms()
    all_vectors = model.wv.get_normed_vectors()
    norm_h, norm_d = avg_vec_h / np.linalg.norm(avg_vec_h), avg_vec_d / np.linalg.norm(avg_vec_d)
    scores = (np.dot(all_vectors, norm_h) + 1) / (np.dot(all_vectors, norm_d) + 1)
    
    lexicon = []
    for i, word in enumerate(vocab):
        if scores[i] >= intensity: lexicon.append({'ngram': word, 'polarity': 'hawkish', 'score': scores[i]})
        elif scores[i] <= (1/intensity): lexicon.append({'ngram': word, 'polarity': 'dovish', 'score': scores[i]})
    return pd.DataFrame(lexicon)

# 5. 톤 점수 산출
def tone_worker(tokens_str):
    if pd.isna(tokens_str): return 0
    tokens = str(tokens_str).split(',')
    features = tokens + master_mpck.ngramize(tokens)
    h = sum(1 for f in features if f in h_lex_set)
    d = sum(1 for f in features if f in d_lex_set)
    return 1 if h > d else (-1 if d > h else 0)

def run_tone_scoring(df, lexicon_df, method_name):
    print(f'\n[{method_name}] 전 기간 톤 점수 산출 중...')
    h, d = set(lexicon_df[lexicon_df['polarity'] == 'hawkish']['ngram']), set(lexicon_df[lexicon_df['polarity'] == 'dovish']['ngram'])
    num_procs = multiprocessing.cpu_count()
    with Pool(processes=num_procs, initializer=init_worker, initargs=(h, d)) as pool:
        results = list(tqdm(pool.imap(tone_worker, df['tokens'], chunksize=1000), total=len(df), desc=f"{method_name} Scoring"))
    return results

# 실제 실행부
if __name__ == "__main__":
    news_files = ['chosun_news_tokenize.csv', 'donga_news_tokenize.csv', 'khan_news_tokenize.csv']
    call_rate_file = 'call_rate.csv'

    # 1. 데이터 로드 및 라벨링
    df_news = prepare_news(news_files)
    df_rate = prepare_call_rate_future(call_rate_file, df_news['date'].min(), df_news['date'].max())
    df_merged = pd.merge(df_news, df_rate[['date', 'label']], on='date', how='inner')

    # 학습 데이터는 25년 12월 전 데이터까지
    test_start = '2025-12-01'
    df_train = df_merged[df_merged['date'] < test_start].copy()
    df_test = df_merged[df_merged['date'] >= test_start].copy()
    print(f'\n데이터 분할 완료: Train({len(df_train)}건), Test({len(df_test)}건)')

    # 2. [학습 데이터만 사용] 사전 구축
    # 시장 접근법 사전 구축
    m_lex = build_market_lexicon(df_train)
    m_lex.to_csv('chosun_donga_khan_market_approach_dict.csv', index=False, encoding='utf-8-sig')
    
    # 사전 접근법 사전 구축
    l_lex = build_lexical_lexicon(df_train)
    l_lex.to_csv('chosun_donga_khan_lexical_approach_dict.csv', index=False, encoding='utf-8-sig')

    # 3. [전체 데이터 적용] 학습된 사전으로 톤 점수 산출
    df_merged['m_sent_label'] = run_tone_scoring(df_merged, m_lex, "Market")
    df_merged['l_sent_label'] = run_tone_scoring(df_merged, l_lex, "Lexical")

    # 4. 일별 지수 집계
    def get_bok_index(group):
        h, d = sum(group == 1), sum(group == -1)
        return (h - d) / (h + d) if (h + d) > 0 else 0

    final_daily = df_merged.groupby('date').agg({'m_sent_label': get_bok_index, 'l_sent_label': get_bok_index, 'label': 'first'}).reset_index()
    final_daily.columns = ['date', 'tone_market', 'tone_lexical', 'rate_label']
    
    final_output = 'chosun_donga_khan_tone_score.csv'
    final_daily.to_csv(final_output, index=False, encoding='utf-8-sig')