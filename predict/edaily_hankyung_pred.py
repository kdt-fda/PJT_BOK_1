import pandas as pd
import statsmodels.api as sm

def run_regression_analysis(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 학습 데이터 (2025년 11월 이전)
    train_data = df[df['date'] < '2025-12-01'].dropna(subset=['label'])
    
    # 독립변수 X (시장 Tone Score, 어휘 Tone Score)
    X = train_data[['tone_market', 'tone_lexical']]
    # 뉴스 외에도 금리에 영향을 주는 기본 요인들을 잡아내기 위해 추가하는 통계적 장치
    X = sm.add_constant(X)
    # 종속변수 y (30일 뒤 금리 변화 라벨)
    y = train_data['label']

    # OLS 모델 학습
    model = sm.OLS(y, X).fit()
    
    # 통계 결과 출력
    print("\n" + "="*60)
    print('[OLS Regression Analysis Summary]')
    print("="*60)
    print(model.summary())

    # 2025년 12월 데이터로 1월 금리 예측
    test_data = df[df['date'] >= '2025-12-01'].copy()
    if test_data.empty:
        print("12월 테스트 데이터가 없습니다.")
        return None
    
    X_test = sm.add_constant(test_data[['tone_market', 'tone_lexical']], has_constant='add')
    
    # 예측값 계산
    test_data['pred_value'] = model.predict(X_test)
    
    # 날짜별 평균 예측값 산출 (날짜별로 하나의 시장 Tone Score와 어휘적 Tone Score 1개씩 나옴)
    daily_pred = test_data.groupby('date')[['tone_market', 'tone_lexical', 'pred_value']].mean().reset_index()
    
    # 최종 예측 방향 판정
    avg_pred_score = daily_pred['pred_value'].mean()
    result_text = "인상(매파)" if avg_pred_score > 0.05 else ("인하(비둘기파)" if avg_pred_score < -0.05 else "동결(중립)")
    
    print("\n" + "="*60)
    print(f"   [2026년 1월 금리 예측 최종 보고]")
    print("="*60)
    print(f" * 12월 평균 예측 지수: {avg_pred_score:.4f}")
    print(f" * 최종 예측 방향: {result_text}")
    print("="*60)

    return daily_pred


if __name__ == "__main__":
# OLS 회귀 분석 및 1월 금리 예측 실행
    regression_report = run_regression_analysis('edaily_hankyung_tone_score.csv')

    # 예측 결과 저장
    if regression_report is not None:
        regression_report.to_csv('prediction_report_202512.csv', index=False, encoding='utf-8-sig')
        print('\n[성공] 모든 분석이 완료되었습니다.')


# 정리
"""
실제 예측 => 최소자승법(OLS) 회귀 모델
'오늘의 뉴스 Tone Score'랑 '30일 후의 금리 변화(Label)'를 쌍으로 매칭하여 모델을 학습시킴.
(뉴스 톤이 실제 금리 결정에 대해 약 1개월의 선행성을 가진다는 논문의 전제)
독립변수(X): 시장 접근법 톤 점수(tone_market), 어휘적 접근법 톤 점수(tone_lexical)
종속변수(y): 과거의 실제 금리 변화 라벨 (-1, 0, 1)
과거 데이터를 분석하여 가중치(회귀 계수)를 도출함.
학습된 모델에 테스트 데이터인 '2025년 12월의 뉴스 Tone Score'를 입력함.
모델은 과거 패턴을 근거로 12월 뉴스의 영향력이 미칠 한 달 뒤(2026년 1월)의 금리 변화 라벨을 예측함.
최종적으로 12월 한 달간 산출된 일별 예측값의 평균이 0.05보다 크면 '1월 금리 인상(매파)'으로, 
-0.05보다 작으면 '1월 금리 인하(비둘기파)'로 최종 예측함. (0.05 값은 금리 기준인 0.03보다 조금 더 엄격한 0.05를 적용)
"""

# 사용한 라이브러리들, 추후에 requirements.txt로 옮기기
"""
pandas
numpy
gensim
ekonlpy
tqdm
statsmodels
beautifulsoup4
requests
kss
"""


seeds_h = ['높/VA','팽창/NNG','인상/NNG','매파/NNG','성장/NNG','투기/NNG';'억제/NNG','상승/NNG','인플레이션/NNG';'압력/NNG','증가/NNG','위험/NNG';'선호/NNG','상회/NNG','물가/NNG';'상승/NNG','과열/NNG','금리/NNG';'상승/NNG','확장/NNG','상방/NNG';'압력/NNG','긴축/NNG','변동성/NNG';'감소/NNG','흑자/NNG','채권/NNG';'가격/NNG';'하락/NNG','견조/NNG','요금/NNG';'인상/NNG','낙관/NNG','부동산/NNG';'가격/NNG';'상승/NNG','상향/NNG']

seeds_d = ['낮/VA','축소/NNG','인하/NNG','비둘기/NNG','둔화/NNG','약화/NNG','하락/NNG','회복/NNG';'못하/VX','감소/NNG','위험/NNG';'회피/NNG','하회/NNG','물가/NNG';'하락/NNG','위축/NNG','금리/NNG';'하락/NNG','침체/NNG','하방/NNG';'압력/NNG','완화/NNG','변동성/NNG';'확대/NNG','적자/NNG','채권/NNG';'가격/NNG';'상승/NNG','부진/NNG','요금/NNG';'인하/NNG','비관/NNG','부동산/NNG';'가격/NNG';'하락/NNG','하향/NNG']

seeds_h = ['낙관/NNG','변동성/NNG','감소/NNG';'위험/NNG','선호/NNG';'매파/NNG','부동산/NNG';'과열/NNG';'억제/NNG','부동산/NNG';'과열/NNG','과열/NNG';'우려/NNG','과열/NNG';'억제/NNG','과열/NNG';'막/VV','경기/NNG';'과열/NNG','부동산/NNG';'과열/NNG';'우려/NNG','경기/NNG';'과열/NNG';'우려/NNG','가격/NNG';'억제/NNG','투자/NNG';'과열/NNG','부동산/NNG';'가격/NNG';'억제/NNG';'경기/NNG','과열/NNG','억제/NNG';'과열/NNG','조정/NNG';'인플레이션/NNG','긴축/NNG';'경기/NNG','과열/NNG','막/VV';'경제/NNG','과열/NNG';'긴축/NNG','압력/NNG';'과열/NNG','방지/NNG']

seeds_d = ['비관/NNG';'요금/NNG','인하/NNG';'적자/NNG';'비둘기/NNG';'둔화/NNG','경기/NNG','침체/NNG';'경기/NNG','침체/NNG','빠지/VV';'약화/NNG','경기/NNG','침체/NNG';'경기/NNG','침체/NNG';'침체/NNG','빠지/VV';'침체/NNG','가능성/NNG','높/VA';'경기/NNG','침체','국면/NNG','빠지/VV';'침체/NNG','경기/NNG','침체/NNG';'둔화/NNG','경기/NNG','침체/NNG';'경기/NNG','침체/NNG','빠지/VV','않/VX';'이미/MAG','침체/NNG';'깊/VA','침체/NNG';'침체/NNG','빠지/VV','우려/NNG';'침체','국면/NNG','빠지/VV';'이미/MAG','경기/NNG','침체/NNG';'침체/NNG','취약/NNG';'경제/NNG','침체/NNG','빠지/VV';'침체/NNG','높/NNG']
