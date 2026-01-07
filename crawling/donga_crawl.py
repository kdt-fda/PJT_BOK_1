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

def crawl_news_donga(url):
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')

        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "제목없음"

        date_match = re.search(r'\d{4}[.\-]\d{2}[.\-]\d{2}', res.text)
        date = date_match.group().replace(".", "-") if date_match else "날짜없음"

        article_section = soup.find('section', {'class': 'news_view'}) or \
                          soup.find('div', {'class': 'article_view'}) or \
                          soup.find('div', {'class': 'news_text'}) or \
                          soup.find('div', {'id': 'article_txt'})

        if article_section:
            # 광고, 관련기사 박스, 스크립트 제거
            target_tags = ['figure', 'script', 'style', 'div', 'iframe', 'button']
            for tag in article_section.find_all(target_tags):
                if tag.name == 'div' and 'article_txt' in tag.get('class', []):
                    continue
                tag.decompose()
            
            content = article_section.get_text(separator="\n", strip=True)
            return [date, f"{title}\n\n{content}"]
            
    except Exception:
        return None
    return None

# 실제 실행부
if __name__ == "__main__":
    all_urls = []
    pages = 833

    # 1. URL 목록 수집
    print('동아일보 URL 목록 수집 시작')
    for page in range(1, pages + 1):
        start_num = (page - 1) * 10 + 1
        base_url = f'https://www.donga.com/news/search?p={start_num}&query=%ED%95%9C%EA%B5%AD%EC%9D%80%ED%96%89+%EA%B8%88%EB%A6%AC&check_news=91&sorting=2&search_date=5&v1=20150101&v2=20251230&more=1'
        
        try:
            res = requests.get(base_url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            datas = soup.find_all('h4')
            if not datas: break
            
            for data in datas:
                link_tag = data.find('a')
                if link_tag and link_tag.get('href'):
                    all_urls.append(link_tag['href'])
            
            if page % 100 == 0:
                print(f"[{page}/{pages}] 페이지 URL 수집 완료 (누적: {len(all_urls)}개)")
        except:
            continue

    # 중복 URL 제거
    all_urls = list(dict.fromkeys(all_urls))
    print(f'\n총 {len(all_urls)}개의 고유 URL 확보, 본문 수집 시작')

    # 2. 멀티스레딩 활용
    docs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(crawl_news_donga, url): url for url in all_urls}
        
        for i, future in enumerate(as_completed(future_to_url)):
            result = future.result()
            if result:
                docs.append(result)
            
            if (i + 1) % 100 == 0:
                print(f'[{i + 1}/{len(all_urls)}] 추출 완료 (현재: {len(docs)})')

    # 3. CSV 파일 저장
    output_csv = 'donga_news.csv'
    df_output = pd.DataFrame(docs, columns=['date', 'full_text'])
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'\n{output_csv} 저장 완료! (총 {len(df_output)}건)')