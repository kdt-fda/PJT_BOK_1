import requests
import pandas as pd

# 1) ECOS API 인증키
API_KEY = "SL3228TC1OVMRYM6LNZK"

# 2) 콜금리 설정
STAT_CODE = "817Y002"      # 일별 시장금리
ITEM_CODE1 = "010101000"   # 콜금리(1일, 전체거래)
CYCLE = "D"
START_DATE = "20050101"
END_DATE   = "20241231"

def fetch_page(start_no: int, end_no: int):
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/"
        f"{API_KEY}/json/kr/{start_no}/{end_no}/"
        f"{STAT_CODE}/{CYCLE}/{START_DATE}/{END_DATE}/{ITEM_CODE1}"
    )
    r = requests.get(url)
    data = r.json()

    if "StatisticSearch" not in data:
        return [], 0

    meta = data["StatisticSearch"]
    rows = meta.get("row", [])
    total_count = int(meta.get("list_total_count", 0))
    return rows, total_count

# 3) 페이지 단위로 전체 수집
page_size = 1000
start_no = 1

all_rows = []
total = None

while True:
    end_no = start_no + page_size - 1
    rows, total_count = fetch_page(start_no, end_no)

    if total is None:
        total = total_count

    if not rows:
        break

    all_rows.extend(rows)

    if len(all_rows) >= total:
        break

    start_no += page_size

# 4) 정리
df = pd.DataFrame(all_rows)

df = df[["TIME", "DATA_VALUE"]].copy()
df["TIME"] = pd.to_datetime(df["TIME"], format="%Y%m%d")
df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")

df = df.sort_values("TIME").reset_index(drop=True)

# 5) CSV 저장
df.to_csv("call_rate.csv", index=False, encoding="utf-8-sig")

