import pandas as pd
import statsmodels.api as sm

# =========================================
# 설정
# =========================================
NEWS_TONE_CSV  = "edaily_hankyung_tone_score.csv"
MACRO_GAPS_CSV = "daily_macro_gaps_linear_interp.csv"

TRAIN_END = "2025-11-30"

THRESH_UP = 0.05
THRESH_DN = -0.05

DATE_COL = "date"
LABEL_COL = "label"
TONE_COLS = ["tone_market", "tone_lexical"]
GAP_COLS  = ["inflation_gap", "output_gap_ip"]

# =========================================
# 1) 로드
# =========================================
def load_news_tone(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df

def load_macro_gaps(path: str) -> pd.DataFrame:
    macro = pd.read_csv(path)
    macro[DATE_COL] = pd.to_datetime(macro[DATE_COL])
    # 필요한 컬럼만
    need = [DATE_COL] + GAP_COLS
    missing = [c for c in need if c not in macro.columns]
    if missing:
        raise ValueError(f"[macro 파일] 필요한 컬럼이 없습니다: {missing}")
    return macro[need].copy()

# =========================================
# 2) 기사 단위 → 날짜 단위로 집계 (tone/label)
#    - 날짜별 기사 여러 개면 mean으로 대표값
# =========================================
def make_daily_from_news(df_news: pd.DataFrame) -> pd.DataFrame:
    need = [DATE_COL] + TONE_COLS + [LABEL_COL]
    missing = [c for c in need if c not in df_news.columns]
    if missing:
        raise ValueError(f"[news 파일] 필요한 컬럼이 없습니다: {missing}")

    daily = (
        df_news
        .groupby(DATE_COL, as_index=False)
        .agg({
            "tone_market": "mean",
            "tone_lexical": "mean",
            LABEL_COL: "mean",   # label이 날짜별로 동일하면 mean=그대로, 혹시 float 되면 아래서 반올림
        })
    )

    # label이 -1/0/1인데 mean으로 float이 됐을 수 있어 반올림해서 복원
    daily[LABEL_COL] = daily[LABEL_COL].round().astype("Int64")
    return daily

# =========================================
# 3) (date 기준) daily tone/label + daily macro gaps merge
#    - 말씀대로 groupby date 하면 gap은 동일하므로 mean 처리(안전장치)
# =========================================
def merge_daily(daily_news: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    # macro도 혹시 중복 날짜가 있으면 mean으로 1일 1행 보장
    macro_daily = (
        macro.groupby(DATE_COL, as_index=False)[GAP_COLS].mean()
    )

    merged = daily_news.merge(macro_daily, on=DATE_COL, how="inner").sort_values(DATE_COL)
    return merged

# =========================================
# 4) OLS 학습 + 예측 (tone을 포함해서 예측)
#    - X: inflation_gap, output_gap_ip, tone_market, tone_lexical
#    - y: label(-1/0/1)
# =========================================
def fit_and_predict(df_daily: pd.DataFrame):
    df_daily = df_daily.dropna(subset=GAP_COLS + TONE_COLS + [LABEL_COL]).copy()

    train = df_daily[df_daily[DATE_COL] < TRAIN_END].copy()
    test  = df_daily[df_daily[DATE_COL] >= TRAIN_END].copy()

    if test.empty:
        raise ValueError("테스트(12월) 구간 데이터가 없습니다. TRAIN_END를 확인하세요.")

    X_cols = GAP_COLS + TONE_COLS

    X_train = sm.add_constant(train[X_cols], has_constant="add")
    y_train = train[LABEL_COL].astype(float)

    model = sm.OLS(y_train, X_train).fit()
    print("\n" + "="*60)
    print("[OLS Summary]  y=label  X=(gaps + tone)")
    print("="*60)
    print(model.summary())

    X_test = sm.add_constant(test[X_cols], has_constant="add")
    test["pred_value"] = model.predict(X_test)

    # 이미 날짜 단위지만, 요청하신 것처럼 groupby(date) + mean 형태 유지
    daily_pred = (
        test.groupby(DATE_COL, as_index=False)[GAP_COLS + TONE_COLS + ["pred_value"]].mean()
    )

    avg_pred = float(daily_pred["pred_value"].mean())
    if avg_pred > THRESH_UP:
        direction = "인상(매파)"
    elif avg_pred < THRESH_DN:
        direction = "인하(비둘기파)"
    else:
        direction = "동결(중립)"

    print("\n" + "="*60)
    print("[2026년 1월 금리 방향 예측(12월 톤 기반)]")
    print("="*60)
    print(f"* 12월 평균 예측 지수: {avg_pred:.4f}")
    print(f"* 최종 예측 방향: {direction}")
    print("="*60)

    return model, daily_pred

# =========================================
# 5) 실행
# =========================================
if __name__ == "__main__":
    # (1) 기사 단위 tone/label -> 날짜 단위 집계
    news = load_news_tone(NEWS_TONE_CSV)
    daily_news = make_daily_from_news(news)

    # (2) 날짜 단위 macro gaps 로드
    macro = load_macro_gaps(MACRO_GAPS_CSV)

    # (3) merge 후 최종 daily 패널
    df_daily = merge_daily(daily_news, macro)

    # (4) OLS 학습 + 예측
    model, report = fit_and_predict(df_daily)

    # (5) 저장
    report.to_csv("prediction_report_202512_daily.csv", index=False, encoding="utf-8-sig")
    print("\n[성공] 저장 완료: prediction_report_202512_daily.csv")
