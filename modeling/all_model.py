#수정한 코드
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

# 데이터 로드 및 시장 금리 라벨링
def prepare_data(news_files, call_rate_file, save_file='news_ngram.pkl'):
    if os.path.exists(save_file):
        print(f"\n기존에 생성된 {save_file} 파일을 불러옵니다.")
        return pd.read_pickle(save_file)
    
    news_list = []
    for f in news_files: # 이데일리, 한경 파일 각각 가져옴
        temp = pd.read_csv(f)
        # 문자열 형태의 데이터를 파이썬의 실제 리스트로 변환
        temp['pos_tokens'] = temp['pos_tokens'].apply(ast.literal_eval) # 그냥 가져오면 리스트로 인식을 못함
        news_list.append(temp)
    news_df = pd.concat(news_list) # 데이터 병합
    news_df['date'] = pd.to_datetime(news_df['date']) # 콜금리랑 news의 날짜를 동일하게 맞춰줌
    
    # ngram을 미리 만들기
    print("\n[전처리] N-gram 생성 중...")
    tqdm.pandas()
    news_df['feature_tokens'] = news_df['pos_tokens'].progress_apply(
        lambda doc: [sent + master_mpck.ngramize(sent) for sent in doc]
    )

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
    merged_df = pd.merge(news_df, call_df[['TIME', 'label']], left_on='date', right_on='TIME', how='left')
    merged_df.to_pickle(save_file)
    return merged_df

def count_features(row):
    all_features = [item for sublist in row[1] for item in sublist]
    return row[0], all_features

# 시장 접근법, n_iterations는 bagging용도, min_freq는 논문에서 언급된 15회보다 적은 단어 필터링용, threshold는 논문에서 언급된 매파, 비둘기파 사전 분류 임계값
def build_market_lexicon(df, n_iterations=30, min_freq=15, threshold=1.3):
    print(f"\n[Market Approach] {n_iterations}회 반복 학습(Bagging) 및 사전 구축 중...")
    
    all_scores = {}
    # 라벨이 NaN인 데이터(25.12)는 사전 구축에서 제외
    train_df_all = df.dropna(subset=['label']).copy()
    
    for _ in range(n_iterations):
        # 데이터를 9:1 비율로 무작위 분할
        # 반복할 때마다 다른 샘플이 선택되어 결과의 안정성을 높임 (논문의 Bagging 방식)
        train_sample = train_df_all.sample(frac=0.9)
        raw_data = train_sample[['label', 'feature_tokens']].values.tolist()
        
        h_counts, d_counts = Counter(), Counter()
        total_h, total_d = 0, 0
        
        # 빈도 계산
        for label, feature_docs in raw_data:
            all_features = [token for sent in feature_docs for token in sent]
            if label == 1: # 매파
                h_counts.update(all_features)
                total_h += len(all_features)
            elif label == -1: # 비둘기파
                d_counts.update(all_features)
                total_d += len(all_features)

        # 이번 회차의 나이브 베이즈 기반 극성 점수 계산
        vocab = set(h_counts.keys()) | set(d_counts.keys())
        for ngram in vocab:
            # 라플라스 스무딩 적용 (데이터가 없는 경우 분모가 0이 되는 것을 방지)
            p_h = (h_counts[ngram] + 1) / (total_h + len(vocab))
            p_d = (d_counts[ngram] + 1) / (total_d + len(vocab))
            score = p_h / p_d
            
            # all_scores 없으면 생성 있으면 점수 추가
            if ngram not in all_scores:
                all_scores[ngram] = []
            all_scores[ngram].append(score)

    # 30회 반복 점수의 평균 산출 및 강도 임계값 적용
    lexicon = []
    total_counts = Counter()
    for _, feature_docs in train_df_all[['label', 'feature_tokens']].values:
        total_counts.update([token for sent in feature_docs for token in sent])

    for ngram, scores in all_scores.items():
        # 최소 빈도 15회 미만 제외
        if total_counts[ngram] < min_freq:
            continue
            
        avg_score = np.mean(scores)
        
        # 논문의 시장 접근법 강도 임계값 1.3 적용 
        # 1.3보다 크면 매파, 1/1.3(=약 0.77)보다 작으면 비둘기파 
        if avg_score >= threshold:
            lexicon.append({'ngram': ngram, 'polarity': 'hawkish', 'score': avg_score})
        elif avg_score <= (1/threshold):
            lexicon.append({'ngram': ngram, 'polarity': 'dovish', 'score': avg_score})
            
    return pd.DataFrame(lexicon)

# 어휘적 접근법
def build_lexical_lexicon(df, n_iterations=50, top_n=3000, filter_top_k=150):
    sentences = [sent for doc in df['feature_tokens'] for sent in doc] # N-gram 포함 토큰
    print("\n[Lexical Approach] Word2Vec 학습 중...")
    # 벡터 차원 300, 주변 단어 범위 5, 최소 개수 15 (논문 값 그대로 사용)
    # gensim 버전에 따라 히위 버전은 size=300으로 수정 필요
    model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=15, workers=4, epochs=30)
    vocab = model.wv.index_to_key
    
    # 단어별 빈도수 확인
    word_counts = {word: model.wv.get_vecattr(word, "count") for word in vocab}
    # 빈도수 내림차순 정렬
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    # 상위 k개 단어는 '불용어(Stopwords)'로 간주하여 제거
    stop_words = set(sorted_words[:filter_top_k])
    
    print(f"\n[전처리] 전체 단어 {len(vocab)}개 중 빈도 상위 {filter_top_k}개 단어를 사전 후보에서 제외합니다.")
    print(f"제외 예시: {sorted_words[:10]}") # 어떤 단어가 빠지는지 확인용
    
    # 논문의 Table 5 기반 씨앗 단어
    seeds_h = [
    '높/VA', '팽창/NNG', '인상/NNG', '매파/NNG', '성장/NNG', 
    '투기/NNG;억제/NNG', '상승/NNG', '인플레이션/NNG;압력/NNG', '증가/NNG', 
    '위험/NNG;선호/NNG', '상회/NNG', '물가/NNG;상승/NNG', '과열/NNG', 
    '금리/NNG;상승/NNG', '확장/NNG', '상방/NNG;압력/NNG', '긴축/NNG', 
    '변동성/NNG;감소/NNG', '흑자/NNG', '채권/NNG;가격/NNG;하락/NNG', 
    '견조/NNG', '요금/NNG;인상/NNG', '낙관/NNG', 
    '부동산/NNG;가격/NNG;상승/NNG', '상향/NNG'
    ]
    seeds_d = [
    '낮/VA', '축소/NNG', '인하/NNG', '비둘기/NNG', '둔화/NNG', 
    '약화/NNG', '하락/NNG', '회복/NNG;못하/VX', '감소/NNG', 
    '위험/NNG;회피/NNG', '하회/NNG', '물가/NNG;하락/NNG', '위축/NNG', 
    '금리/NNG;하락/NNG', '침체/NNG', '하방/NNG;압력/NNG', '완화/NNG', 
    '변동성/NNG;확대/NNG', '적자/NNG', '채권/NNG;가격/NNG;상승/NNG', 
    '부진/NNG', '요금/NNG;인하/NNG', '비관/NNG', 
    '부동산/NNG;가격/NNG;하락/NNG', '하향/NNG'
    ]
    
    all_seeds = set(seeds_h) | set(seeds_d)
    # 사전에 등록할 후보 단어 리스트 (불용어 제외 근데 seeds word면 넣음)
    candidate_vocab = [w for w in vocab if (w not in stop_words) or (w in all_seeds)]
    
    # seed words 중에서 실제 뉴스에 나온 seed word로 업데이트, 이 단어 목록으로 유사도를 측정해서 사전에 추가하게 됨
    seeds_h, seeds_d = [s for s in seeds_h if s in vocab], [s for s in seeds_d if s in vocab]
    
    if not seeds_h or not seeds_d:
        raise ValueError('seed words가 없습니다. 단어를 교체하세요')
    
    # 부트스트래핑 및 점수 계산: 50회 반복 전파 로직 구현 
    # 각 단어별 매파/비둘기파 점수 누적용 딕셔너리
    final_scores = {word: {'h': 0.0, 'd': 0.0} for word in candidate_vocab}
    
    print(f"\n[Lexical Approach] {n_iterations}회 반복 전파(Bootstrapping) 중...")
    for _ in tqdm(range(n_iterations)):
        # 씨앗 단어의 80%를 무작위로 선택 (통계적 안정성 확보하기 위함, 유사도의 정확한 측정용)
        sample_h = np.random.choice(seeds_h, size=max(1, int(len(seeds_h)*0.8)), replace=False)
        sample_d = np.random.choice(seeds_d, size=max(1, int(len(seeds_d)*0.8)), replace=False)
        
        for word in candidate_vocab:
            # 코사인 유사도 평균 계산 후 지수 함수(exp) 적용 (음수 방지)
            # 논문의 SentProp 확률 전파 개념을 유사도 공간에서 모사한 부분
            sim_h = np.mean([model.wv.similarity(word, s) for s in sample_h])
            sim_d = np.mean([model.wv.similarity(word, s) for s in sample_d])
            
            final_scores[word]['h'] += sim_h
            final_scores[word]['d'] += sim_d
            
    lexicon = []
    for word, scores in final_scores.items():
        avg_h = scores['h'] / n_iterations
        avg_d = scores['d'] / n_iterations
        score = avg_h - avg_d
        lexicon.append({'ngram': word, 'score': score})
    
    df_lex = pd.DataFrame(lexicon)
    
    # 매파 단어 추출 (점수 높은 순)
    hawkish_words = df_lex.nlargest(top_n, 'score').copy()
    hawkish_words['polarity'] = 'hawkish'
    
    # 비둘기파 단어 추출 (점수 낮은 순)
    dovish_words = df_lex.nsmallest(top_n, 'score').copy()
    dovish_words['polarity'] = 'dovish'
    
    # 결과 합치기
    final_lexicon = pd.concat([hawkish_words, dovish_words])

    print(f"\n매파 단어: {len(hawkish_words)}개, 비둘기파 단어: {len(dovish_words)}개 구축 완료")
    print(f"매파 점수 범위: {hawkish_words['score'].min():.4f} ~ {hawkish_words['score'].max():.4f}")
    print(f"비둘기파 점수 범위: {dovish_words['score'].min():.4f} ~ {dovish_words['score'].max():.4f}")

    return final_lexicon[['ngram', 'polarity', 'score']]

# Tone Score 계산
def tone_worker(feature_tokens):
    global h_lexicon_set, d_lexicon_set
    sent_labels = []
    for features in feature_tokens:
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
    with Pool(processes=4, initializer=init_worker, initargs=(h_set, d_set)) as pool:
        results = list(tqdm(pool.imap(tone_worker, df['feature_tokens'], chunksize=50), total=len(df)))
    return results


# 메인
if __name__ == "__main__":
    news_files = ['edaily_news_all_tokenize.csv', 'hankyung_news_all_tokenize.csv', 'debenture_tokenize.csv', 'donga_news_tokenize.csv', 'hani_news_tokenize.csv',
                  'khan_news_tokenize.csv', 'minutes_tokenize.csv', 'chosun_news_tokenize.csv']
    all_data = prepare_data(news_files, 'call_rate.csv', 'news_ngram.pkl')

    # 학습 데이터 (2012.01 ~ 2025.11) / 테스트 데이터 (2025.12)
    train_data = all_data[all_data['date'] < '2025-12-01'].copy()

    # 시장 접근법 사전 구축 (25년 12월 제외 데이터)
    market_lex_file = 'all_market_approach_dict.csv'
    if os.path.exists(market_lex_file):
        print(f"\n기존에 생성된 시장 접근법 사전({market_lex_file})을 불러옵니다.")
        m_lex = pd.read_csv(market_lex_file)
    else:
        print("\n시장 접근법 사전을 새로 구축합니다.")
        m_lex = build_market_lexicon(train_data)
        m_lex.to_csv(market_lex_file, index=False, encoding='utf-8-sig')
    
    # 어휘적 접근법 사전 구축 (25년 12월 제외 데이터)
    lexical_lex_file = 'all_lexical_approach_dict.csv'
    if os.path.exists(lexical_lex_file):
        print(f"\n기존에 생성된 어휘적 접근법 사전({lexical_lex_file})을 불러옵니다.")
        l_lex = pd.read_csv(lexical_lex_file)
    else:
        print("\n어휘적 접근법 사전을 새로 구축합니다.")
        l_lex = build_lexical_lexicon(train_data)
        l_lex.to_csv(lexical_lex_file, index=False, encoding='utf-8-sig')

    # Tone Score 계산 (전체 데이터로 함)
    all_data['tone_market'] = run_multiprocess_tone(all_data, m_lex, "Market")
    all_data['tone_lexical'] = run_multiprocess_tone(all_data, l_lex, "Lexical")
    
    # 날짜별 시장 접근법, 어휘 접근법 톤 점수 평균화
    daily_sentiment = all_data.groupby('date')[['tone_market', 'tone_lexical']].mean().reset_index()
    
    # 날짜별 뉴스 건수 확인
    daily_counts = all_data.groupby('date').size().reset_index(name='news_count')
    daily_sentiment = pd.merge(daily_sentiment, daily_counts, on='date')
    
    # 금리 label도 한개로 통합, 어차피 같은 날짜 label은 동일하므로 first로 함
    daily_label = all_data.groupby('date')['label'].first().reset_index()
    daily_sentiment = pd.merge(daily_sentiment, daily_label, on='date')

    # 결과 저장
    daily_sentiment.to_csv('all_tone_score.csv', index=False, encoding='utf-8-sig')
    print("\n완료되었습니다.")
