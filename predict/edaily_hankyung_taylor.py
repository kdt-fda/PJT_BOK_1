import pandas as pd
import statsmodels.api as sm

# =========================================
# 설정
# =========================================
NEWS_TONE_CSV  = "edaily_hankyung_tone_score.csv"
MACRO_GAPS_CSV = "daily_macro_gaps_linear_interp.csv"

# ✅ 11월까지 학습, 12월만 예측
TRAIN_END   = "2025-11-30"   # train: <= TRAIN_END
TEST_START  = "2025-12-01"
TEST_END    = "2025-12-31"

THRESH_UP = 0.05
THRESH_DN = -0.05

DATE_COL  = "date"
LABEL_COL = "label"

# ✅ 시장접근법만 사용
TONE_COLS = ["tone_market"]

# 테일러 준칙 변수(통제변수)
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

    need = [DATE_COL] + GAP_COLS
    missing = [c for c in need if c not in macro.columns]
    if missing:
        raise ValueError(f"[macro 파일] 필요한 컬럼이 없습니다: {missing}")

    return macro[need].copy()

# =========================================
# 2) 기사 단위 → 날짜 단위 집계 (tone/label)
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
            "tone_market": "mean",  # ✅ market tone만
            LABEL_COL: "mean",
        })
    )

    # label(-1/0/1) 복원
    daily[LABEL_COL] = daily[LABEL_COL].round().astype("Int64")
    return daily

# =========================================
# 3) merge (12월 tone가 안 잘리도록 left merge + ffill)
# =========================================
def merge_daily(daily_news: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    macro_daily = macro.groupby(DATE_COL, as_index=False)[GAP_COLS].mean()

    merged = (
        daily_news
        .merge(macro_daily, on=DATE_COL, how="left")  # ✅ inner → left
        .sort_values(DATE_COL)
    )

    # ✅ macro가 11월까지만 있어도 12월을 예측해야 하므로, 가장 최근 값으로 채움
    merged[GAP_COLS] = merged[GAP_COLS].ffill()

    return merged

# =========================================
# 4) 모델 학습 + 예측 (12월만 테스트로)
# =========================================
def fit_predict_with_cols(df_daily: pd.DataFrame, X_cols: list[str], model_name: str):
    df_daily = df_daily.sort_values(DATE_COL).copy()

    # train/test 분리 (요청하신 구조)
    train = df_daily[df_daily[DATE_COL] <= TRAIN_END].copy()
    test  = df_daily[(df_daily[DATE_COL] >= TEST_START) & (df_daily[DATE_COL] <= TEST_END)].copy()

    # 학습은 라벨이 있어야 함
    train = train.dropna(subset=X_cols + [LABEL_COL]).copy()

    if test.empty:
        raise ValueError("테스트(12월) 구간 데이터가 없습니다. TEST_START/TEST_END 또는 원천 데이터를 확인하세요.")

    # 테스트는 라벨 없어도 예측 가능하지만, X는 필요하니 X 결측만 제거
    test = test.dropna(subset=X_cols).copy()
    if test.empty:
        raise ValueError("테스트(12월) 구간에서 X가 전부 결측입니다. merge/ffill 로직을 확인하세요.")

    # OLS 학습
    X_train = sm.add_constant(train[X_cols], has_constant="add")
    y_train = train[LABEL_COL].astype(float)

    model = sm.OLS(y_train, X_train).fit()

    print("\n" + "="*70)
    print(f"[{model_name}] OLS Summary")
    print("="*70)
    print(model.summary())

    # 예측
    X_test = sm.add_constant(test[X_cols], has_constant="add")
    test = test.copy()
    test["pred_value"] = model.predict(X_test)

    # 날짜별 평균(일별 예측치)
    daily_pred = test.groupby(DATE_COL, as_index=False)[X_cols + ["pred_value"]].mean()

    # 12월 평균 예측지수로 방향 판정(선택)
    avg_pred = float(daily_pred["pred_value"].mean())
    if avg_pred > THRESH_UP:
        direction = "인상(매파)"
    elif avg_pred < THRESH_DN:
        direction = "인하(비둘기파)"
    else:
        direction = "동결(중립)"

    print("\n" + "-"*70)
    print(f"[{model_name}] 12월 Test Avg Pred: {avg_pred:.4f}  =>  {direction}")
    print("-"*70)

    metrics = {
        "model": model_name,
        "R2": float(model.rsquared),
        "Adj_R2": float(model.rsquared_adj),
        "AIC": float(model.aic),
        "BIC": float(model.bic),
        "N": int(model.nobs),
    }

    return model, daily_pred, metrics

# =========================================
# 5) 실행
# =========================================
if __name__ == "__main__":
    purpose = (
        "본 분석은 테일러 준칙 변수(인플레이션 갭, 산출갭)만 포함한 기본 모형과 "
        "시장접근법 톤 점수(tone_market)만을 추가한 확장 모형을 각각 OLS로 추정한 뒤, "
        "설명력(R²/Adj.R²) 및 정보기준(AIC/BIC)을 비교하여 tone_market의 추가적 설명력 여부를 검증한다. "
        "또한 11월까지 학습한 뒤 12월 구간을 테스트로 예측한다."
    )
    print("\n" + "="*70)
    print("[분석 목적]")
    print("="*70)
    print(purpose)
    print("="*70)

    # 데이터 준비
    news = load_news_tone(NEWS_TONE_CSV)
    daily_news = make_daily_from_news(news)

    macro = load_macro_gaps(MACRO_GAPS_CSV)
    df_daily = merge_daily(daily_news, macro)

    print("\n" + "="*70)
    print("[df_daily 날짜 범위]")
    print("="*70)
    print(df_daily[DATE_COL].min(), "~", df_daily[DATE_COL].max())
    print("="*70)

    # Model A: Taylor only
    X_taylor_only = GAP_COLS
    model_A, pred_A, metrics_A = fit_predict_with_cols(
        df_daily, X_taylor_only, "MODEL A (Taylor only: gaps)"
    )

    # Model B: Taylor + market tone only
    X_taylor_market = GAP_COLS + ["tone_market"]
    model_B, pred_B, metrics_B = fit_predict_with_cols(
        df_daily, X_taylor_market, "MODEL B (Taylor + market tone)"
    )

    # 비교표
    comp = pd.DataFrame([metrics_A, metrics_B])
    print("\n" + "="*70)
    print("[설명력 비교: Taylor only vs Taylor+MarketTone]")
    print("="*70)
    print(comp.to_string(index=False))
    print("="*70)

    # 예측 저장 (12월 날짜 기준으로 합치기)
    out = pred_A[[DATE_COL, "pred_value"]].rename(columns={"pred_value": "pred_taylor_only"}).merge(
        pred_B[[DATE_COL, "pred_value"]].rename(columns={"pred_value": "pred_taylor_market"}),
        on=DATE_COL,
        how="outer"
    ).sort_values(DATE_COL)

    out.to_csv("prediction_report_compare_market_only_daily.csv", index=False, encoding="utf-8-sig")
    print("\n[성공] 저장 완료: prediction_report_compare_market_only_daily.csv")
