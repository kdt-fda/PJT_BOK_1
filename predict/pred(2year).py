import pandas as pd
import statsmodels.api as sm

NEWS_TONE_CSV  = "all_tone_score.csv"
MACRO_GAPS_CSV = "daily_macro_gaps_linear_interp.csv"

TRAIN_END   = "2023-12-31"   # train: <= TRAIN_END
TEST_START  = "2024-01-01"
TEST_END    = "2025-11-30"

THRESH_UP = 0.05
THRESH_DN = -0.05

DATE_COL  = "date"
LABEL_COL = "label"

# 시장접근법만 사용
TONE_COLS = ["tone_market"]

# 테일러 준칙 변수(통제변수)
GAP_COLS  = ["inflation_gap", "gdp_gap"]

# 로드
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

# 기사 단위 → 날짜 단위 집계 (tone/label)
def make_daily_from_news(df_news: pd.DataFrame) -> pd.DataFrame:
    need = [DATE_COL] + TONE_COLS + [LABEL_COL]
    missing = [c for c in need if c not in df_news.columns]
    if missing:
        raise ValueError(f"[news 파일] 필요한 컬럼이 없습니다: {missing}")

    daily = (
        df_news
        .groupby(DATE_COL, as_index=False)
        .agg({
            "tone_market": "mean",  # market tone만
            LABEL_COL: "mean",
        })
    )

    # label(-1/0/1) 복원
    daily[LABEL_COL] = daily[LABEL_COL].round().astype("Int64")
    return daily

# merge (12월 tone가 안 잘리도록 left merge + ffill)
def merge_daily(daily_news: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    macro_daily = macro.groupby(DATE_COL, as_index=False)[GAP_COLS].mean()

    merged = (
        daily_news
        .merge(macro_daily, on=DATE_COL, how="left")
        .sort_values(DATE_COL)
    )

    # macro가 11월까지만 있어도 12월을 예측해야 하므로, 가장 최근 값으로 채움
    merged[GAP_COLS] = merged[GAP_COLS].ffill()

    return merged


# 모델 학습 + 평가 (정확도 산출 로직 추가)
def fit_predict_with_cols(df_daily: pd.DataFrame, X_cols: list[str], model_name: str):
    df_daily = df_daily.sort_values(DATE_COL).copy()

    # train/test 분리
    train = df_daily[df_daily[DATE_COL] <= TRAIN_END].copy()
    test  = df_daily[(df_daily[DATE_COL] >= TEST_START) & (df_daily[DATE_COL] <= TEST_END)].copy()

    # 학습 및 테스트 모두 라벨(정답)이 있어야 검증 가능하므로 결측 제거
    train = train.dropna(subset=X_cols + [LABEL_COL]).copy()
    test_eval = test.dropna(subset=X_cols + [LABEL_COL]).copy() # 평가용 (정답 있는 데이터)

    if train.empty:
        raise ValueError(f"학습 데이터가 없습니다. (TRAIN_END: {TRAIN_END})")
    
    # OLS 학습
    X_train = sm.add_constant(train[X_cols], has_constant="add")
    y_train = train[LABEL_COL].astype(float)

    model = sm.OLS(y_train, X_train).fit()

    print("\n" + "="*70)
    print(f"[{model_name}] OLS Summary (Train: ~{TRAIN_END})")
    print("="*70)
    print(model.summary())

    # 테스트 셋(2024~2025)에 대한 예측 및 정확도 검증
    if not test_eval.empty:
        X_test = sm.add_constant(test_eval[X_cols], has_constant="add")
        
        # 예측값 생성
        test_eval["pred_value"] = model.predict(X_test)
        
        # 방향성 결정 (1: 인상, 0: 동결, -1: 인하)
        # 예측치(pred_value)가 THRESH_UP보다 크면 1, DN보다 작으면 -1, 아니면 0
        test_eval["pred_dir"] = test_eval["pred_value"].apply(
            lambda x: 1 if x > THRESH_UP else (-1 if x < THRESH_DN else 0)
        )
        
        # 실제 값(Label)도 정수형으로 확실히 변환 (이미 되어있을 수 있지만 안전장치)
        test_eval["actual_dir"] = test_eval[LABEL_COL].round().astype(int)

        # 정확도 계산 (Directional Accuracy)
        correct_count = (test_eval["pred_dir"] == test_eval["actual_dir"]).sum()
        total_count = len(test_eval)
        accuracy = correct_count / total_count if total_count > 0 else 0

        print("\n" + "-"*70)
        print(f"[{model_name}] 2024-2025 Test Performance")
        print(f" - Accuracy (방향성 정확도): {accuracy:.2%} ({correct_count}/{total_count})")
        print(f" - MAE (평균 절대 오차): {(test_eval['pred_value'] - test_eval[LABEL_COL]).abs().mean():.4f}")
        print("-"*70)
        
        # 리턴용 데이터 (시각화 등을 위해)
        daily_pred = test_eval[[DATE_COL] + X_cols + [LABEL_COL, "pred_value", "pred_dir", "actual_dir"]]
    else:
        print(f"[{model_name}] 경고: 테스트 기간에 정답(Label) 데이터가 없어 정확도를 계산할 수 없습니다.")
        daily_pred = pd.DataFrame()

    metrics = {
        "model": model_name,
        "Train_R2": float(model.rsquared),
        "Test_Accuracy": accuracy if not test_eval.empty else None, # 테스트 정확도 추가
        "AIC": float(model.aic),
        "N_Train": int(model.nobs),
        "N_Test": len(test_eval)
    }

    return model, daily_pred, metrics

# 실행
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

    # 비교표 출력
    comp = pd.DataFrame([metrics_A, metrics_B])
    print("\n" + "="*70)
    print("[모델 성능 비교: Train R² vs Test Accuracy]")
    print("="*70)
    print(comp.to_string(index=False))
    print("="*70)

    # 두 모델의 예측 결과를 날짜 기준으로 병합
    if not pred_A.empty and not pred_B.empty:
        out = pred_A[[DATE_COL, LABEL_COL, "pred_value"]].rename(
            columns={"pred_value": "pred_taylor", LABEL_COL: "actual_label"}
        ).merge(
            pred_B[[DATE_COL, "pred_value"]].rename(columns={"pred_value": "pred_with_tone"}),
            on=DATE_COL,
            how="inner"
        ).sort_values(DATE_COL)

        filename = "result_2024_2025_validation.csv"
        out.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"\n[성공] 검증 결과 저장 완료: {filename}")
        print(" -> 이 파일을 열어서 2024~2025년 동안 모델이 금리 방향을 얼마나 잘 따라갔는지 확인하세요.")