import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 데이터 로드
file_path = 'all_tone_score.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date']) # 날짜 형식 변환
    print(f"데이터 로드 완료: {len(df)}건")
else:
    print(f"{file_path} 파일이 없습니다.")

# 기초 통계량 출력 (평균, 표준편차, 최대/최소 등)
print("\n[기초 통계량 요약]")
print(df[['tone_market', 'tone_lexical']].describe())

# 0점(Zero Score) 개수 확인
zeros_market = (df['tone_market'] == 0).sum()
zeros_lexical = (df['tone_lexical'] == 0).sum()
print("\n[0점 개수 확인]")
print(f"시장 접근법(Market) 0점 개수: {zeros_market}개")
print(f"어휘 접근법(Lexical) 0점 개수: {zeros_lexical}개")

plt.figure(figsize=(14, 6))

# 시장 접근법 분포
plt.subplot(1, 2, 1)
sns.histplot(df['tone_market'], bins=50, kde=True, color='blue')
plt.title('Market Approach Tone Score Distribution')
plt.xlabel('Tone Score (Hawkish vs Dovish)')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--', linewidth=1) # 0점 기준선

# 어휘 접근법 분포
plt.subplot(1, 2, 2)
sns.histplot(df['tone_lexical'], bins=50, kde=True, color='green')
plt.title('Lexical Approach Tone Score Distribution')
plt.xlabel('Tone Score (Hawkish vs Dovish)')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()

# 월별 시계열 추이
# 일별 데이터는 너무 많아 보기 힘들므로, 월별 평균을 냄
df_monthly = df.set_index('date').resample('M')[['tone_market', 'tone_lexical']].mean()

plt.figure(figsize=(14, 6))
plt.plot(df_monthly.index, df_monthly['tone_market'], label='Market Approach', color='blue', alpha=0.7)
plt.plot(df_monthly.index, df_monthly['tone_lexical'], label='Lexical Approach', color='green', alpha=0.7)

plt.title('Monthly Average Tone Score Trend (2012 ~ 2025)')
plt.xlabel('Date')
plt.ylabel('Average Tone Score')
plt.axhline(0, color='red', linestyle='--', linewidth=1) # 0점 기준선
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# 정리
"""
양수 : 매파 (Hawkish)

물가 상승을 걱정하거나, 금리를 인상해야 한다는 의견이 우세한 날

점수가 1에 가까울수록 금리 인상 신호가 강력한 매파 성향

음수 : 비둘기파 (Dovish)

경기가 안 좋으니 부양해야 한다거나, 금리를 인하하거나 동결해야 한다는 의견이 우세한 날

점수가 -1에 가까울수록 경기 침체가 심각하며, 금리 인하의 목소리가 지배적인 강력한 비둘기파 성향

0 : 중립 (Neutral)

매파와 비둘기파의 의견이 팽팽하게 맞서거나, 특별한 방향성 없이 단순 사실만 나열된 경우
"""