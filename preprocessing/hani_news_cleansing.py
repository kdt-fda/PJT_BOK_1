import re
import pandas as pd

# 1) 데이터 로드
df = pd.read_csv(
    "hani_news.csv",
    encoding="utf-8-sig"
)

print(f"원본 기사 수: {len(df)}")

# 2) 기자 이름 제거 함수
def remove_reporter(text: str) -> str:
    if not isinstance(text, str):
        return ""

    patterns = [
        r"\(?[가-힣]{2,4}\s*기자\)?",
        r"\(?[가-힣]{2,4}\s*선임기자\)?",
        r"\(?[가-힣]{2,4}\s*특파원\)?",
        r"[가-힣]{2,4}\s*기자\s*=",
    ]

    for p in patterns:
        text = re.sub(p, " ", text)

    return text


# 3) 한글만 남기기 + 공통 잡음 제거 (숫자, 마침표 유지)
def clean_korean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 기자명 제거
    text = remove_reporter(text)

    # 저작권 / 안내 문구 제거
    noise_patterns = [
        r"ⓒ\s*한겨레.*",
        r"무단\s*전재.*",
        r"재배포\s*금지.*",
        r"기사\s*본문\s*내용.*",
        r"Copyright.*",
    ]

    for p in noise_patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)

    # ✅ 한글 + 숫자 + 공백 + 마침표(.)만 남기기
    text = re.sub(r"[^가-힣0-9\s\.]", " ", text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 4) 클렌징 적용
df["full_text"] = df["full_text"].apply(clean_korean_text)

# 빈 문장 제거
before = len(df)
df = df[df["full_text"].str.len() > 10].reset_index(drop=True)
after = len(df)
print(f"빈/무의미 기사 제거: {before - after}건")

# 5) 중복 제거
before = len(df)
df = df.drop_duplicates(subset=["full_text"]).reset_index(drop=True)
after = len(df)
print(f"중복 기사 제거: {before - after}건")

# 6) 저장
df[["date", "full_text"]].to_csv(
    "news_hani_filtered.csv",
    index=False,
    encoding="utf-8-sig"
)

print("✅ 저장 완료: news_hani_filtered.csv")
print(f"최종 기사 수: {len(df)}")
