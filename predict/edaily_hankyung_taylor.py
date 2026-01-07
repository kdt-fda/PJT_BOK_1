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
    need = [DATE_COL] + GAP_COLS
    missing = [c for c in need if c not in macro.columns]
    if missing:
        raise ValueError(f"[macro 파일] 필요한 컬럼이 없습니다: {missing}")
    return macro[need].copy()

# =========================================
# 2) 기사 단위 → 날짜 단위로 집계 (tone/label)
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
            LABEL_COL: "mean",
        })
    )
    daily[LABEL_COL] = daily[LABEL_COL].round().astype("Int64")
    return daily

# =========================================
# 3) merge
# =========================================
def merge_daily(daily_news: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    macro_daily = macro.groupby(DATE_COL, as_index=False)[GAP_COLS].mean()
    merged = daily_news.merge(macro_daily, on=DATE_COL, how="inner").sort_values(DATE_COL)
    return merged

# =========================================
# 4) 모델 학습 + 예측
# =========================================
def fit_predict_with_cols(df_daily: pd.DataFrame, X_cols: list[str], model_name: str):
    df_daily = df_daily.dropna(subset=X_cols + [LABEL_COL]).copy()

    train = df_daily[df_daily[DATE_COL] < TRAIN_END].copy()
    test  = df_daily[df_daily[DATE_COL] >= TRAIN_END].copy()

    if test.empty:
        raise ValueError("테스트 구간 데이터가 없습니다. TRAIN_END를 확인하세요.")

    X_train = sm.add_constant(train[X_cols], has_constant="add")
    y_train = train[LABEL_COL].astype(float)

    model = sm.OLS(y_train, X_train).fit()

    print("\n" + "="*70)
    print(f"[{model_name}] OLS Summary")
    print("="*70)
    print(model.summary())

    X_test = sm.add_constant(test[X_cols], has_constant="add")
    test = test.copy()
    test["pred_value"] = model.predict(X_test)

    daily_pred = (
        test.groupby(DATE_COL, as_index=False)[X_cols + ["pred_value"]].mean()
    )

    avg_pred = float(daily_pred["pred_value"].mean())
    if avg_pred > THRESH_UP:
        direction = "인상(매파)"
    elif avg_pred < THRESH_DN:
        direction = "인하(비둘기파)"
    else:
        direction = "동결(중립)"

    print("\n" + "-"*70)
    print(f"[{model_name}] Test Avg Pred: {avg_pred:.4f}  =>  {direction}")
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
    # (0) 연구 목적
    purpose = (
        "본 분석은 테일러 준칙에 포함되는 거시경제 변수(인플레이션 갭, 산출갭)만을 포함한 기본 모형과 "
        "여기에 뉴스 텍스트로부터 산출한 통화정책 톤 점수를 추가한 확장 모형을 각각 OLS로 추정한 뒤, "
        "두 모형의 설명력(R², Adj.R²) 및 정보기준(AIC/BIC)을 비교함으로써 톤 변수의 추가적 설명력 여부를 "
        "검증하는 것을 목적으로 한다."
    )
    print("\n" + "="*70)
    print("[분석 목적]")
    print("="*70)
    print(purpose)
    print("="*70)

    # (1) 데이터 준비
    news = load_news_tone(NEWS_TONE_CSV)
    daily_news = make_daily_from_news(news)

    macro = load_macro_gaps(MACRO_GAPS_CSV)

    df_daily = merge_daily(daily_news, macro)

    print("\n" + "="*70)
    print("[df_daily 날짜 범위]")
    print("="*70)
    print(df_daily[DATE_COL].min(), "~", df_daily[DATE_COL].max())
    print("="*70)

    # (2) Model A: Taylor only (톤 미포함)
    X_taylor_only = GAP_COLS
    model_A, pred_A, metrics_A = fit_predict_with_cols(
        df_daily, X_taylor_only, "MODEL A (Taylor only: gaps)"
    )

    # (3) Model B: Taylor + Tone (톤 포함)
    X_taylor_tone = GAP_COLS + TONE_COLS
    model_B, pred_B, metrics_B = fit_predict_with_cols(
        df_daily, X_taylor_tone, "MODEL B (Taylor + Tone: gaps + tone)"
    )

    # (4) 설명력 비교 표 출력
    comp = pd.DataFrame([metrics_A, metrics_B])
    print("\n" + "="*70)
    print("[설명력 비교: Taylor only vs Taylor+Tone]")
    print("="*70)
    print(comp.to_string(index=False))
    print("="*70)

    # (5) 예측 결과 저장(날짜 기준으로 합치기)
    out = pred_A[[DATE_COL, "pred_value"]].rename(columns={"pred_value": "pred_taylor_only"}).merge(
        pred_B[[DATE_COL, "pred_value"]].rename(columns={"pred_value": "pred_taylor_tone"}),
        on=DATE_COL,
        how="outer"
    ).sort_values(DATE_COL)

    out.to_csv("prediction_report_compare_daily.csv", index=False, encoding="utf-8-sig")
    print("\n[성공] 저장 완료: prediction_report_compare_daily.csv")
