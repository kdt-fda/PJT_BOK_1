# hankyung 한국경제
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor # 멀티스레딩

url = "https://search.hankyung.com/search/news?query=%ED%95%9C%EA%B5%AD%EC%9D%80%ED%96%89+%EA%B8%88%EB%A6%AC&sort=DATE%2FDESC%2CRANK%2FDESC&period=DATE&area=ALL&sdate=2012.01.01&edate=2025.12.31&exact=&include=&except=&hk_only=n"
headers = {"User-Agent": "Mozilla/5.0"}

hankyung = []

def fetch_article(article_url):
    try: # 검색 창 결과 내의 각 기사의 url에 들어감
        res = requests.get(article_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 날짜, 제목, 본문 추출
        date_elem = soup.select_one('span.txt-date')
        date_text = date_elem.get_text(strip=True)[:10].replace('.', '-') if date_elem else "날짜 없음" # 시분초 버리고 날짜만 추출
        
        headline = soup.select_one('h1.headline')
        title = headline.get_text(strip=True) if headline else "제목 없음" # 제목 없으면 제목 없음
        
        body = soup.select_one('#articletxt')
        content = body.get_text(' ', strip=True) if body else "본문 없음" # 본문 없으면 본문 없음
        
        if title or content:
            full_text = f"{title} {content}".strip()
            return [date_text, full_text] # [날짜, 본문] 형태로 반환
        
    except Exception as e:
        return None
    return None

page = 1
try:
    while True:
        real_url = f"{url}&page={page}"
        print(f"\n[페이지 {page}] 크롤링 중...")
        
        res = requests.get(real_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        articles = soup.select('ul.article > li div.txt_wrap > a')
        
        if not articles:
            break
        
        # 페이지에 있는 모든 기사의 url을 저장
        url_list = [tag['href'] for tag in articles]

        with ThreadPoolExecutor(max_workers=10) as executor: # 페이지에 기사가 10개씩 있어서 10으로 지정해서 스레딩
            results = list(executor.map(fetch_article, url_list))
        
        # None 제외하고 결과 추가
        hankyung.extend([r for r in results if r])
        print(f"현재까지 {len(hankyung)}건 수집 완료")
        
        page += 1
        time.sleep(0.3)

except KeyboardInterrupt: # 중간에 중지시 거기까지 저장
    print("\n[중단] 사용자에 의해 멈췄습니다. 현재까지 수집된 데이터를 저장합니다...")


print(f"\n총 {len(hankyung)}건의 기사를 수집했습니다.")
df = pd.DataFrame(hankyung, columns=['date', 'full_text'])

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

# CSV 파일 저장 (헤더 없이 저장하려면 header=False 추가)
df.to_csv("hankyung_news_all.csv", index=False, header=False, encoding='utf-8-sig')
print("파일 저장 완료: hankyung_news_all.csv")