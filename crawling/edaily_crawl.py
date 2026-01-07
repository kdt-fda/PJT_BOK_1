# edaily 이데일리
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time

def get_news_data(start_date, end_date, edaily):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    page = 1
    
    while 1:
        url = f"https://www.edaily.co.kr/search/news/?source=total&keyword=%ed%95%9c%ea%b5%ad%ec%9d%80%ed%96%89+%ea%b8%88%eb%a6%ac&start={start_date}&end={end_date}&sort=latest&date=pick&page={page}"
        
        try:
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            news_items = soup.select("#newsList .newsbox_04")
            
            if not news_items:
                break
            
            for item in news_items:
                date_text = ''
                date_elem = item.select_one("div.author_category")
                if date_elem:
                    date_text = date_elem.contents[0].strip().replace('.', '-')
                    
                lis = item.select("ul.newsbox_texts li")
                # 제목이랑 본문 합침
                full_text = " ".join([li.get_text(strip=True) for li in lis])
                
                if full_text:
                    edaily.append([date_text, full_text])
            
            print(f"[{start_date}~{end_date}] {page}페이지 완료...")
            page += 1
            time.sleep(0.3) 
            
        except Exception as e:
            print(f"오류 발생: {e}")
            break
            
    return edaily

start_dt = datetime(2012, 1, 1)
end_dt = datetime(2025, 12, 31)
current_dt = start_dt
edaily = []

try:
    while current_dt < end_dt:
        interval_end = current_dt + timedelta(days=182) # 간격을 6개월, 약 182일로 지정
        if interval_end > end_dt:
            interval_end = end_dt
        
        s_str = current_dt.strftime("%Y%m%d") # 이데일리의 url 날짜 포맷
        e_str = interval_end.strftime("%Y%m%d")
        
        print(f"\n>>> 수집 중: {s_str} ~ {e_str}")
        edaily = get_news_data(s_str, e_str, edaily)
        
        current_dt = interval_end + timedelta(days=1) # interval_end로 주면 맨 마지막 날짜 겹치므로 +1

except KeyboardInterrupt: # 중간에 중지시 거기까지 저장
    print("\n[중단] 사용자에 의해 멈췄습니다. 현재까지 수집된 데이터를 저장합니다...")

print(f"\n총 {len(edaily)}건의 기사를 수집했습니다.")
df = pd.DataFrame(edaily, columns=['date', 'full_text'])

df['date_dt'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
df = df.sort_values(by='date_dt', ascending=True) # 날짜 오름차순 정렬
df = df.drop(columns=['date_dt']) # 임시로 만들었던 열 삭제

df['full_text'] = df['full_text'].str.replace(r'[\n\r]+', ' ', regex=True).str.strip() # csv 저장할 때 줄바꿈 공백 없애서 줄 수 맞추기

# 중복 기사 제거
initial_len = len(df)
df = df.drop_duplicates(subset=['full_text']).reset_index(drop=True)
final_len = len(df)

if initial_len != final_len:
    print(f"중복된 기사 {initial_len - final_len}건을 제외했습니다.")

df.to_csv("edaily_news_all.csv", index=False, header=False, encoding="utf-8-sig")
print("파일 저장 완료: edaily_news_all.csv")