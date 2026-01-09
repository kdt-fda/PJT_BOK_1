import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

url = "https://search.hankyung.com/search/news?query=%ED%95%9C%EA%B5%AD%EC%9D%80%ED%96%89+%EA%B8%88%EB%A6%AC&sort=DATE%2FDESC%2CRANK%2FDESC&period=DATE&area=ALL&sdate=2012.01.01&edate=2025.12.31&exact=&include=&except=&hk_only=n"
headers = {"User-Agent": "Mozilla/5.0"}

hankyung = []

def fetch_article(article_url):
    try:  # 검색 창 결과 내의 각 기사의 url에 들어감
        res = requests.get(article_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 날짜, 제목, 본문 추출
        date_elem = soup.select_one('span.txt-date')
        date_text = date_elem.get_text(strip=True)[:10].replace('.', '-') if date_elem else "날짜 없음" # 시분초 버리고 날짜만 추출
        
        headline = soup.select_one('h1.headline')
        title = headline.get_text(strip=True) if headline else "제목 없음" # 제목 없으면 제목 없음
        
        body = soup.select_one('#articletxt')
        content = body.get_text(' ', strip=True) if body else "본문 없음" # 본문 없으면 본문 없음
        
        if title or content:
            return [date_text, f"{title} {content}".strip()] # [날짜, 본문] 형태로 반환
    except:
        return None
    return None

def fetch_page(page_num): # 페이지 하나의 전체 기사 수집하는 부분
    real_url = f"{url}&page={page_num}"
    print(f"\n[페이지 {page_num}] 크롤링 중...")
    try:
        res = requests.get(real_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        articles = soup.select('ul.article > li div.txt_wrap > a')
        
        if not articles:
            return []

        page_data = []
        for tag in articles:
            result = fetch_article(tag['href'])
            if result:
                page_data.append(result)
        return page_data
    except:
        return []

current_page = 1 # 크롤링 시작할 페이지
chunk_size = 10 # 한 번에 처리할 페이지 수

try:
    while True:
        print(f"\n[페이지 {current_page} ~ {current_page + chunk_size - 1}] 병렬 수집 중...")
        
        # 10개 페이지 번호 생성
        pages = range(current_page, current_page + chunk_size)
        
        # 10개의 스레드를 사용하여 페이지별로 크롤링 수행
        with ThreadPoolExecutor(max_workers=chunk_size) as executor:
            batch_results = list(executor.map(fetch_page, pages))
        
        # 결과 합치기
        found_data = False
        for page_items in batch_results:
            if page_items:
                hankyung.extend(page_items)
                found_data = True
        
        print(f"현재까지 총 {len(hankyung)}건 수집 완료")
        
        # 만약 이번 뭉치(10페이지)에서 데이터가 하나도 없었다면 종료
        if not found_data:
            print("더 이상 가져올 데이터가 없습니다.")
            break
            
        current_page += chunk_size
        time.sleep(0.5) # 서버 부하 방지를 위한 짧은 휴식

except KeyboardInterrupt: # 중간에 중지시 거기까지 저장
    print("\n[중단] 사용자에 의해 중지되었습니다.")

print(f"\n총 {len(hankyung)}건의 기사를 수집했습니다.")
df = pd.DataFrame(hankyung, columns=['date', 'full_text'])

df['date_dt'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
df = df.sort_values(by='date_dt', ascending=True).drop(columns=['date_dt']) # 날짜 오름차순 정렬, 임시로 만들었던 열 삭제

df['full_text'] = df['full_text'].str.replace(r'[\n\r]+', ' ', regex=True).str.strip() # csv 저장할 때 줄바꿈 공백 없애서 줄 수 맞추기

# 중복 기사 제거
initial_len = len(df)
df = df.drop_duplicates(subset=['full_text']).reset_index(drop=True)
final_len = len(df)

if initial_len != final_len:
    print(f"중복된 기사 {initial_len - final_len}건을 제외했습니다.")

# CSV 파일 저장 (헤더 없이 저장하기 위해 header=False 작성)
df.to_csv("hankyung_news_all.csv", index=False, header=False, encoding='utf-8-sig')
print(f"\n최종 {len(df)}건 저장 완료: hankyung_news_all.csv")
