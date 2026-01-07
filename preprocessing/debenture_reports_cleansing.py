import pandas as pd
import re
import os

df_report=pd.read_csv('raw_debenture_reports_text.csv')
df_page=pd.read_csv('raw_naver_debenture_page_text.csv')

#reports text
col = "text"   # 필요 시 컬럼명 변경

s = df_report[col].fillna("").astype(str)

# 1) 괄호 안 내용까지 제거 ((), [], {}, <>)
s = s.str.replace(r"\([^)]*\)", " ", regex=True)
s = s.str.replace(r"\[[^\]]*\]", " ", regex=True)
s = s.str.replace(r"\{[^}]*\}", " ", regex=True)
s = s.str.replace(r"\<[^>]*\>", " ", regex=True)

# 2) 이메일 제거 (영문 제거 전에 명시적으로 처리)
s = s.str.replace(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    " ",
    regex=True
)

# 3) 숫자 제거 (정수, 소수 포함)
s = s.str.replace(r"\d+", " ", regex=True)

# 4) 영어 + 특수기호 + 문장부호 전부 제거
#    → 한글과 공백만 남김
s = s.str.replace(r"[^가-힣\s]+", " ", regex=True)

# 5) 한국어 특정 단어/문장: "년", "월", "일" 제거 (단어 경계가 애매하므로 공백 기준으로 제거)
#    "2025년" 같은 건 숫자 제거 후 "년"만 남는 경우가 많아서 여기서 제거됨
s = s.str.replace(r"(?<![가-힣])(?:년|월|일)(?![가-힣])", " ", regex=True)
# 참고: 만약 "내년" 같은 단어에서 '년'까지 지우고 싶지 않다면 위처럼 경계를 둔 방식이 안전함

# 6) 공백 정리
s = (
    s.str.replace(r"\s+", " ", regex=True)
     .str.strip()
)

df_report[col] = s

#page_text

col = "page_text"   # 필요 시 컬럼명 변경

s = df_page[col].fillna("").astype(str)

# 1) 괄호 안 내용까지 제거 ((), [], {}, <>)
s = s.str.replace(r"\([^)]*\)", " ", regex=True)
s = s.str.replace(r"\[[^\]]*\]", " ", regex=True)
s = s.str.replace(r"\{[^}]*\}", " ", regex=True)
s = s.str.replace(r"\<[^>]*\>", " ", regex=True)

# 2) 이메일 제거 (영문 제거 전에 명시적으로 처리)
s = s.str.replace(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    " ",
    regex=True
)

# 3) 숫자 제거 (정수, 소수 포함)
s = s.str.replace(r"\d+", " ", regex=True)

# 4) 영어 + 특수기호 + 문장부호 전부 제거
#    → 한글과 공백만 남김
s = s.str.replace(r"[^가-힣\s]+", " ", regex=True)

# 5) 한국어 특정 단어/문장: "년", "월", "일" 제거 (단어 경계가 애매하므로 공백 기준으로 제거)
#    "2025년" 같은 건 숫자 제거 후 "년"만 남는 경우가 많아서 여기서 제거됨
s = s.str.replace(r"(?<![가-힣])(?:년|월|일)(?![가-힣])", " ", regex=True)
# 참고: 만약 "내년" 같은 단어에서 '년'까지 지우고 싶지 않다면 위처럼 경계를 둔 방식이 안전함

# 6) 공백 정리
s = (
    s.str.replace(r"\s+", " ", regex=True)
     .str.strip()
)

df_report[col] = s

# 저장할 컬럼명 (본인 DF에 맞게 수정)
nid_col = "nid"
text_col = "text"   # 클렌징 적용한 컬럼

# 1) nid + text만 뽑아서 저장용 DF 만들기
df_out = df_report[[nid_col, text_col]].copy()

# 2) (선택) 중복 nid가 있으면 마지막 것만 남기기
# df_out = df_out.drop_duplicates(subset=[nid_col], keep="last")

# 3) 저장
out_path = os.path.join(r"C:\Users\User\Desktop\kdt\pjt_bok_1", "report_text_clean_v1.csv")  # BASE_DIR 없으면 원하는 경로로
df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

print("saved:", out_path, "| rows:", len(df_out))

df_out_report = df_report[[nid_col, text_col]].copy()


# 저장할 컬럼명 (본인 DF에 맞게 수정)
date_col= "date"
org_col="org"
title_col="title"
source_raw_col="source_raw"
nid_col = "nid"
page_text_col = "page_text"   # 클렌징 적용한 컬럼

# 1) nid + text만 뽑아서 저장용 DF 만들기
df_out = df_page[[nid_col,date_col,org_col,title_col,source_raw_col,page_text_col]].copy()

# 2) (선택) 중복 nid가 있으면 마지막 것만 남기기
# df_out = df_out.drop_duplicates(subset=[nid_col], keep="last")

# 3) 저장
out_path = os.path.join(r"C:\Users\User\Desktop\kdt\pjt_bok_1", "page_text_clean_v1.csv")  # BASE_DIR 없으면 원하는 경로로
df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

print("saved:", out_path, "| rows:", len(df_out))

df_out_page = df_page[[nid_col,date_col,org_col,title_col,source_raw_col,page_text_col]].copy()

# date 컬럼: "2025.12.31" -> "2025-12-31"
df_out_page["date"] = (
    df_out_page["date"]
    .astype(str)
    .str.strip()
    .pipe(pd.to_datetime, format="%Y.%m.%d", errors="coerce")
    .dt.strftime("%Y-%m-%d")
)

out_path = os.path.join(r"C:\Users\User\Desktop\kdt\pjt_bok_1", "page_text_clean_v1.csv")

df_out_page.to_csv(
    out_path,
    index=False,
    encoding="utf-8-sig"
)

print("saved:", out_path)

# nid 타입 통일 (중요)
df_out_page["nid"] = df_out_page["nid"].astype(str)
df_out_report["nid"] = df_out_report["nid"].astype(str)

# 병합 (page 기준)
df_merged = (
    df_out_page
    .merge(
        df_out_report,
        on="nid",
        how="left"   # page는 유지, report 없으면 text는 NaN
    )
)

print(df_out_page.columns.tolist())
print(df_out_report.columns.tolist())

df_merged = df_merged[
    ["nid", "date", "org", "title", "source_raw", "page_text", "text"]
]


col = "page_text"

s = df_merged[col].fillna("").astype(str)

# 1) 괄호 안 내용까지 제거 ((), [], {}, <>)
s = s.str.replace(r"\([^)]*\)", " ", regex=True)
s = s.str.replace(r"\[[^\]]*\]", " ", regex=True)
s = s.str.replace(r"\{[^}]*\}", " ", regex=True)
s = s.str.replace(r"\<[^>]*\>", " ", regex=True)

# 2) 이메일 제거 (혹시 남아있는 경우 대비)
s = s.str.replace(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    " ",
    regex=True
)

# 3) 숫자 제거 (정수, 소수 포함)
s = s.str.replace(r"\d+", " ", regex=True)

# 4) 영어 + 특수기호 + 문장부호 전부 제거
#    → 한글과 공백만 남김
s = s.str.replace(r"[^가-힣\s]+", " ", regex=True)

# 5) 공백 정리
s = (
    s.str.replace(r"\s+", " ", regex=True)
     .str.strip()
)

df_merged[col] = s

import os

out_path = os.path.join(r"C:\Users\User\Desktop\kdt\pjt_bok_1", "cleansed_debenture_reports.csv")

df_merged.to_csv(
    out_path,
    index=False,
    encoding="utf-8-sig"
)

print("saved:", out_path, "| rows:", len(df_merged))
