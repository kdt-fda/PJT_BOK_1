import pandas as pd
import statsmodels.api as sm

TRAIN_END, TEST_RANGE = "2025-11-30", ("2025-12-01", "2025-12-31")
GAP_COLS = ["inflation_gap", "gdp_gap"]

df_news = pd.read_csv("all_tone_score_call_rate.csv", parse_dates=['date'])
df_macro = pd.read_csv("daily_macro_gaps_linear_interp.csv", parse_dates=['date'])

df = pd.merge(df_news, df_macro[['date'] + GAP_COLS], on='date', how='left').sort_values('date')
df[GAP_COLS] = df[GAP_COLS].ffill()


def run_analysis(data, x_features, model_name):
    train = data[data['date'] <= TRAIN_END].dropna(subset=x_features + ['DATA_VALUE'])
    test = data[(data['date'] >= TEST_RANGE[0]) & (data['date'] <= TEST_RANGE[1])].dropna(subset=x_features)

    # Z-score 표준화 추가
    all_cols = x_features + ['DATA_VALUE']
    train_mean = train[all_cols].mean()
    train_std = train[all_cols].std()

    # 학습 데이터 표준화
    train_scaled = (train[all_cols] - train_mean) / train_std
    # 테스트 데이터 표준화 (학습 데이터의 평균/표준편차 기준으로 함)
    test_scaled = (test[x_features] - train_mean[x_features]) / train_std[x_features]

    # 모델 학습 (표준화된 데이터 사용)
    X_train = sm.add_constant(train_scaled[x_features], has_constant='add')
    model = sm.OLS(train_scaled['DATA_VALUE'], X_train).fit()
    
    print(f"\n{'='*30}")
    print(f"[{model_name}] 상세 결과")
    print(f"{'='*30}")
    print(model.summary())

    X_test = sm.add_constant(test_scaled, has_constant='add')
    test_pred_scaled = model.predict(X_test)
    
    # 역표준화로 원래 금리 복원
    test_pred = (test_pred_scaled * train_std['DATA_VALUE']) + train_mean['DATA_VALUE']
    
    print(f"\n[{model_name}] R2: {model.rsquared:.4f} | 1월 예측평균: {test_pred.mean():.4f}%")
    return model, test_pred


# Model A: 거시지표만 / Model B: 거시지표 + 뉴스 톤 => y는 1달 뒤 콜금리
res_A = run_analysis(df, GAP_COLS, "Taylor Only")
res_B = run_analysis(df, GAP_COLS + ["tone_market"], "Taylor + Tone")


report = pd.DataFrame({
    'date': df.loc[(df['date'] >= TEST_RANGE[0]) & (df['date'] <= TEST_RANGE[1]), 'date'],
    'pred_taylor': res_A[1],
    'pred_with_tone': res_B[1]
})

report.to_csv("final_prediction_call_rate.csv", index=False, encoding="utf-8-sig")

print("\n[완료] 결과가 'final_prediction_call_rate.csv'에 저장되었습니다.")