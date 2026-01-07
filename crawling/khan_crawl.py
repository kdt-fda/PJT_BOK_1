import requests
from bs4 import BeautifulSoup
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# 설정 및 전역 변수
headers = {
    'User-Agent': 'Mozilla/5.0'
}

def crawl_news_khan(url):
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')

        title_tags = soup.find_all('h1')
        title = title_tags[1].get_text(strip=True) if len(title_tags) > 1 else title_tags[0].get_text(strip=True)

        date_element = soup.find('div', {'class': 'date'}) or soup.find('span', {'class': 'date'})
        date_match = re.search(r'\d{4}[.\-]\d{2}[.\-]\d{2}', date_element.text) if date_element else None
        date = date_match.group().replace(".", "-") if date_match else "날짜없음"

        # 1번 방법 : content_text 클래스 우선 탐색
        content_elements = soup.find_all('p', {'class': 'content_text'})
        if content_elements:
            full_content = '\n'.join([el.get_text(strip=True) for el in content_elements])
        else:
            # 2번 : 클래스가 없는 경우 본문 영역 탐색
            article_body = soup.find('div', {'class': 'article_txt'}) or \
                           soup.find('article', {'id': 'articleBody'}) or \
                           soup.find('div', {'class': 'art_body'})
            if article_body:
                for tag in article_body.find_all(['script', 'style', 'figure', 'div', 'iframe']):
                    tag.decompose()
                full_content = article_body.get_text(separator="\n", strip=True)
            else:
                full_content = ""

        # 노이즈 제거
        stops = ["뱅크-아이", "무단전재", "기자 =", "기자=", "@khan.co.kr", "ⓒ", "제공=", "출처="]
        for stop in stops:
            if stop in full_content:
                full_content = full_content.split(stop)[0]

        # 이메일 및 URL 제거
        full_content = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', full_content)
        full_content = re.sub(r'http[s]?://\S+', '', full_content)
        
        content = full_content.strip()
        if content:
            return [date, f"{title}\n\n{content}"]
            
    except Exception:
        return None
    return None

# 실제 실행부
if __name__ == "__main__":
    all_urls = []
    pages = 425

    # 1. URL 목록 수집
    print('경향신문 URL 목록 수집 시작')
    for page in range(1, pages + 1):
        base_url = f'https://search.khan.co.kr/?q=%ED%95%9C%EA%B5%AD%EC%9D%80%ED%96%89+%EA%B8%88%EB%A6%AC&media=khan&page={page}&section=0&term=5&startDate=2015-01-01&endDate=2025-12-30&sort=2'
        try:
            res = requests.get(base_url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')

            datas = soup.find_all('a', {'ep_event_area': '검색결과_기사목록'})
            
            for data in datas:
                url = data.get('href')
                if url:
                    all_urls.append(url)
            
            if page % 50 == 0:
                print(f'[{page}/{pages}] 페이지 URL 수집 완료 (누적: {len(all_urls)}개)')
        except:
            continue

    # 중복 URL 제거
    all_urls = list(dict.fromkeys(all_urls))
    print(f'\n총 {len(all_urls)}개의 고유 URL 확보, 본문 수집 시작')

    # 2. 멀티스레딩 활용
    docs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(crawl_news_khan, url): url for url in all_urls}
        
        for i, future in enumerate(as_completed(future_to_url)):
            result = future.result()
            if result:
                docs.append(result)
            
            if (i + 1) % 100 == 0:
                print(f'[{i + 1}/{len(all_urls)}] 추출 완료 (현재: {len(docs)})')

    # 3. CSV 파일 저장
    output_csv = 'khan_news.csv'
    df_output = pd.DataFrame(docs, columns=['date', 'full_text'])
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'\n{output_csv} 저장 완료! (총 {len(df_output)}건)')