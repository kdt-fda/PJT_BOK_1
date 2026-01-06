import pandas as pd
import numpy as np
import ast
from collections import Counter
from gensim.models import Word2Vec
from ekonlpy.sentiment import MPCK
from multiprocessing import Pool
from tqdm import tqdm
import os

# 윈도우 인코딩 오류(cp949)를 방지
os.environ["PYTHONUTF8"] = "1"

# 전역 변수 설정
master_mpck = MPCK()
h_lexicon_set = set()
d_lexicon_set = set()

def init_worker(h_set=None, d_set=None):
    global h_lexicon_set, d_lexicon_set
    if h_set is not None:
        h_lexicon_set = h_set
    if d_set is not None:
        d_lexicon_set = d_set

# 데이터 로드 및 시장 금리 라벨링 (시장 접근법 용도)
def prepare_data(news_files, call_rate_file):
    news_list = []
    for f in news_files: # 이데일리, 한경 파일 각각 가져옴
        temp = pd.read_csv(f)
        # 문자열 형태의 데이터를 파이썬의 실제 리스트로 변환
        temp['pos_tokens'] = temp['pos_tokens'].apply(ast.literal_eval) # 그냥 가져오면 리스트로 인식을 못함
        news_list.append(temp)
    news_df = pd.concat(news_list) # 데이터 병합
    news_df['date'] = pd.to_datetime(news_df['date']) # 콜금리랑 news의 날짜를 동일하게 맞춰줌

    call_df = pd.read_csv(call_rate_file)
    call_df['TIME'] = pd.to_datetime(call_df['TIME']) # 콜금리랑 news의 날짜를 동일하게 맞춰줌
    
    all_dates = pd.date_range(start='2012-01-01', end='2025-12-31')
    call_df = call_df.set_index('TIME').reindex(all_dates)
    call_df = call_df.fillna(method='ffill').fillna(method='bfill').reset_index() # ffill: 이전 영업일 금리로 주말 채우기, bfill: 2012-01-01처럼 시작일에 데이터가 없는 경우 1월 2일 금리로 채우기
    call_df.columns = ['TIME', 'DATA_VALUE']
    
    # 발표일 기준 1개월(30일) 후 콜금리 변화 라벨링
    # 해당 날짜 시점으로부터 30일 이후의 콜금리 값을 해당 날짜 콜금리에서 뺌. 논문 내용처럼 +-0.03 임계값 기준으로 매파, 비둘기파, 0이면 중립으로 나눔 
    call_df['label'] = call_df['DATA_VALUE'].diff(30).shift(-30).apply(
        lambda x: 1 if x >= 0.03 else (-1 if x <= -0.03 else 0)
    )
    
    # 날짜 기준으로 뉴스랑 콜금리를 합침(30일 계산한 그 값으로)
    # 날짜, 토큰화 값, 해당 날짜 기준 1개월 후의 금리 변화(라벨)
    merged_df = pd.merge(news_df, call_df[['TIME', 'label']], left_on='date', right_on='TIME', how='inner')
    return merged_df

def count_features(row):
    all_features = []
    for sent_tokens in row[1]: 
        ngrams = master_mpck.ngramize(sent_tokens)
        all_features.extend(sent_tokens + ngrams)
    return row[0], all_features 

# 시장 접근법, min_freq는 논문에서 언급된 15회보다 적은 단어 필터링용, threshold는 논문에서 언급된 매파, 비둘기파 사전 분류 임계값
def build_market_lexicon(df, min_freq=15, threshold=1.3):
    print("\n[Market Approach] 사전 구축 중...")
    # 라벨이 NaN인 데이터(25.12)는 사전 구축에서 제외
    train_df = df.dropna(subset=['label'])
    raw_data = train_df[['label', 'pos_tokens']].values.tolist()
    
    h_counts, d_counts = Counter(), Counter() # 매파, 비둘기파 단어별 카운트
    total_h, total_d = 0, 0 # 전체 단어 총개수
    with Pool(processes=2, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(count_features, raw_data, chunksize=50), total=len(raw_data)))
        
    for label, features in results: # 매파(1)는 h, 비둘기파(-1)은 d
        if label == 1: h_counts.update(features); total_h += len(features)
        elif label == -1: d_counts.update(features); total_d += len(features)
            
    lexicon = []
    # 나이브 베이즈 확률 및 점수 계산
    for ngram in (set(h_counts.keys()) | set(d_counts.keys())):
        if (h_counts[ngram] + d_counts[ngram]) < min_freq: continue # 15회 미만은 무시
        p_h = h_counts[ngram] / total_h if total_h > 0 else 0 # 매파 뉴스에서 해당 단어가 나올 확률
        p_d = d_counts[ngram] / total_d if total_d > 0 else 0 # 비둘기파 뉴스에서 해당 단어가 나올 확률
        score = p_h / p_d if p_d > 0 else 0 # 이 스코어 값이 1.3 전후로 매파, 비둘기파 사전이 갈리게 됨
        if score >= threshold: lexicon.append({'ngram': ngram, 'polarity': 'hawkish', 'score': score})
        elif score <= (1/threshold): lexicon.append({'ngram': ngram, 'polarity': 'dovish', 'score': score})
    return pd.DataFrame(lexicon)

# 어휘적 접근법, intensity는 논문에서 나온 임계값
def build_lexical_lexicon(df, intensity=1.1):
    sentences = [sent for doc in df['pos_tokens'] for sent in doc] # 기사의 문장별 토큰 리스트 
    print("\n[Lexical Approach] Word2Vec 학습 중...")
    # 벡터 차원 300, 주변 단어 범위 5, 최소 개수 15 (논문 값 그대로 사용)
    # gensim 버전에 따라 상위 버전은 vector_size=300으로 수정 필요
    model = Word2Vec(sentences=sentences, size=300, window=5, min_count=15, workers=1)
    
    # 논문의 Table 5 기반 씨앗 단어
    seeds_h = ['인상/NNG', '상승/NNG', '확대/NNG', '긴축/NNG', '낙관/NNG', '강화/NNG', '과열/NNG', '호조/NNG']
    seeds_d = ['인하/NNG', '하락/NNG', '축소/NNG', '완화/NNG', '비관/NNG', '약화/NNG', '둔화/NNG', '침체/NNG']
    
    vocab = model.wv.index_to_key
    # seed words 중에서 실제 뉴스에 나온 seed word로 업데이트
    seeds_h, seeds_d = [s for s in seeds_h if s in vocab], [s for s in seeds_d if s in vocab]
    
    if not seeds_h or not seeds_d:
        raise ValueError('seed words가 없습니다. 단어를 교체하세요')
    
    lexicon = []
    # 코사인 유사도 및 점수 계산
    for word in tqdm(vocab, desc="유사도 계산중"):
        sim_h = np.mean([model.wv.similarity(word, s) for s in seeds_h])
        sim_d = np.mean([model.wv.similarity(word, s) for s in seeds_d])
        score = (sim_h + 1) / (sim_d + 1) # 이 스코어 값이 1.1 전후로 매파, 비둘기파 사전이 갈리게 됨
        if score >= intensity: lexicon.append({'ngram': word, 'polarity': 'hawkish', 'score': score})
        elif score <= (1/intensity): lexicon.append({'ngram': word, 'polarity': 'dovish', 'score': score})
    return pd.DataFrame(lexicon)

# Tone Score 계산
def tone_worker(pos_tokens):
    global h_lexicon_set, d_lexicon_set
    sent_labels = []
    for sent in pos_tokens:
        features = sent + master_mpck.ngramize(sent) # 기사의 단어와 ngram 추출
        h = sum(1 for f in features if f in h_lexicon_set) # 문장에서 매파 사전의 단어 개수
        d = sum(1 for f in features if f in d_lexicon_set) # 문장에서 비둘기파 사전의 단어 개수
        if h > d: sent_labels.append(1) # 매파 단어가 많으면 매파 라벨
        elif d > h: sent_labels.append(-1) # 비둘기파 단어가 많으면 비둘기파 라벨
    nh, nd = sent_labels.count(1), sent_labels.count(-1) # 매파 문장, 비둘기파 문장 집계
    return (nh - nd) / (nh + nd) if (nh + nd) > 0 else 0 # Tone Score 계산

def run_multiprocess_tone(df, lexicon_df, method_name):
    print(f"\n[{method_name}] Tone Score 계산 중...")
    h_set = set(lexicon_df[lexicon_df['polarity'] == 'hawkish']['ngram']) # 사전을 set 형태로 전환, Tone Score 계산을 위해 사전 분리
    d_set = set(lexicon_df[lexicon_df['polarity'] == 'dovish']['ngram'])
    with Pool(processes=2, initializer=init_worker, initargs=(h_set, d_set)) as pool:
        results = list(tqdm(pool.imap(tone_worker, df['pos_tokens'], chunksize=50), total=len(df)))
    return results


# 메인
if __name__ == "__main__":
    news_files = ['edaily_news_all_tokenize.csv', 'hankyung_news_all_tokenize.csv']
    all_data = prepare_data(news_files, 'call_rate.csv')

    # 학습 데이터 (2012.01 ~ 2025.11) / 테스트 데이터 (2025.12)
    train_data = all_data[all_data['date'] < '2025-12-01'].copy()

    # 시장 접근법 사전 구축 (25년 12월 제외 데이터)
    m_lex = build_market_lexicon(train_data)
    m_lex.to_csv('edaily_hankyung_market_approach_dict.csv', index=False, encoding='utf-8-sig')
    
    # 어휘적 접근법 사전 구축 (25년 12월 제외 데이터)
    l_lex = build_lexical_lexicon(train_data)
    l_lex.to_csv('edaily_hankyung_lexical_approach_dict.csv', index=False, encoding='utf-8-sig')

    # Tone Score 계산 (전체 데이터로 함)
    all_data['tone_market'] = run_multiprocess_tone(all_data, m_lex, "Market")
    all_data['tone_lexical'] = run_multiprocess_tone(all_data, l_lex, "Lexical")

    # 결과 저장
    all_data[['date', 'tone_market', 'tone_lexical', 'label']].to_csv('edaily_hankyung_tone_score.csv', index=False, encoding='utf-8-sig')
    print("\n완료되었습니다.")
    


# 정리
"""
시장 접근법
일단 전체 데이터에 대해 30일 이후의 콜금리랑 해당 날짜의 콜금리랑
비교해서 0.03보다 크면 매파, -0.03보다 작으면 비둘기파, 그 사이면 중립으로 라벨링(label)을 함.
그리고 라벨링된 기사에서 매파 기사, 비둘기파 기사를 분리함.
매파 기사에서 나온 특정 단어랑 비둘기파 기사에서 나온 특정 단어(즉 둘이 같은 단어)가 합해서 15회 미만이면 영향력이 없으므로 무시.
매파 기사 단어 전체에서 특정 단어가 나온 확률 / 비둘기파 기사 단어 전체에서 특정 단어가 나온 확률을 계산했을 때, 1.3 보다 크면 해당 단어를 최종적으로 매파 단어로 분류.
1/1.3보다 작으면 해당 단어를 최종적으로 비둘기파 단어로 분류.

어휘적 접근법
모든 기사의 문장에서 어떤 단어들이 서로 붙어 다니지 분석함.
그렇게 해서 300차원의 공간에 좌표로 찍음.(뜻이 비슷한 단어들이 가까운 곳에 위치하게 됨)
매파, 비둘기파 씨앗 단어를 설정함.
사전에 등록할 단어들과 각각의 씨앗 단어 간의 코사인 유사도를 계산함.
(매파 씨앗 단어와의 유사도 + 1) / (비둘기파 씨앗 단어와의 유사도) + 1)로 계산했을 때, 1.1 보다 크면 매파 단어로 분류, 1/1.1 보다 작으면 비둘기파 단어로 최종 분류.

Tone Score
구축된 '시장 접근법 사전'과 '어휘적 접근법 사전'을 사용하여 개별 기사 및 날짜별 시장 심리를 수치화함.
기사를 구성하는 각 문장 내에 포함된 매파 단어(h)와 비둘기파 단어(d)의 개수를 카운트함.
문장 내 h > d 이면 해당 문장은 '매파 문장(+1)'으로 분류.
문장 내 d > h 이면 해당 문장은 '비둘기파 문장(-1)'으로 분류.
두 개수가 같거나 단어가 없으면 '중립 문장(0)'으로 분류.
(매파 문장 수 - 비둘기파 문장 수) / (매파 문장 수 + 비둘기파 문장 수) 공식을 적용하여 Tone Score를 계산함.
결과값은 -1(극단적 비둘기파)에서 +1(극단적 매파) 사이의 값을 가짐.
"""