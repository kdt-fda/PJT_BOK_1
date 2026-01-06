import requests
import json
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import pandas as pd

# 설정 및 전역 변수
headers = {
    'User-Agent': 'Mozilla/5.0'
}

DATE_RANGES = []

for yy in range(12, 26):
    FIRST_START_DATE = f'20{yy}0101'
    FIRST_END_DATE = f'20{yy}0630'

    DATE_RANGES.append((FIRST_START_DATE, FIRST_END_DATE))

    SECOND_START_DATE = f'20{yy}0701'
    SECOND_END_DATE = f'20{yy}1231'

    DATE_RANGES.append((SECOND_START_DATE, SECOND_END_DATE))

print(DATE_RANGES)

def crawl_news_chosun(url):
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'utf-8'
        html = res.text
        
        # 1번 방법
        pattern = r'Fusion\.globalContent\s*=\s*(\{.*?\});'
        match = re.search(pattern, html)
        if match:
            data = json.loads(match.group(1))

            title = data.get('headlines', {}).get('basic', '제목없음').strip()

            date = data.get('display_date', '날짜없음')[:10]

            elements = data.get('content_elements', [])
  
            paragraphs = [re.sub(r'<[^>]*>', '', el.get('content', '')) 
                          for el in elements if el.get('type') in ['text', 'raw_html']]
            
            content = "\n".join(paragraphs).strip()

            if content:
                return [date, f"{title}\n\n{content}"]

        # 2번 방법
        soup = BeautifulSoup(html, 'html.parser')
        
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "제목없음"

        date_match = re.search(r'\d{4}[.\-]\d{2}[.\-]\d{2}', html)
        date = date_match.group().replace(".", "-") if date_match else "날짜없음"
        
        article_body = soup.find('article') or \
                       soup.find('div', id='news_body_id') or \
                       soup.find('div', class_='article_body') or \
                       soup.find('div', id='article_body') or \
                       soup.find('section', id='articleAll') or \
                       soup.find('div', class_='par')

        if article_body:
            # 불필요한 태그 제거
            target_tags = ['script', 'style', 'iframe', 'textarea', 'header', 'footer', 'button', 'figure']
            for tag in article_body.find_all(target_tags):
                tag.decompose()
            
            # 광고/저작권/관련박스 제거
            for div in article_body.find_all('div', class_=['art_ad', 'art_copyright', 're_box', 'art_ad_wrap']):
                div.decompose()

            content = article_body.get_text(separator="\n", strip=True)
            return [date, f"{title}\n\n{content}"]

    except Exception:
        return None
    return None

# 실제 실행부
if __name__ == "__main__":
    all_urls = []
    
    # 1. URL 목록 수집
    print('조선일보 URL 목록 수집 시작')
    for s_date, e_date in DATE_RANGES:
        print(f'기간 {s_date} ~ {e_date} 처리 중...')
        for page in range(400):
            search_url = f'https://search-gateway.chosun.com/nsearch?query=금리&page={page}&size=10&sort=2&r=direct&s={s_date}&e={e_date}'
            try:
                res = requests.get(search_url, headers=headers, timeout=10)
                items = res.json().get('content_elements', [])
                if not items: break
                
                for item in items:
                    if item.get('site_url'):
                        all_urls.append(item['site_url'])
            except:
                continue

    # 중복 URL 제거
    all_urls = list(dict.fromkeys(all_urls))
    print(f'\n총 {len(all_urls)}개의 고유 URL 확보, 본문 수집 시작')

    # 2. 멀티스레딩 활용
    docs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(crawl_news_chosun, url): url for url in all_urls}
        
        for i, future in enumerate(as_completed(future_to_url)):
            result = future.result()
            if result:
                docs.append(result)
            
            if (i + 1) % 100 == 0:
                print(f'[{i + 1}/{len(all_urls)}] 추출 완료 (현재: {len(docs)})')

    # 3. CSV 파일 저장
    output_csv = 'chosun_news.csv'
    df_output = pd.DataFrame(docs, columns=['date', 'full_text'])
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'\n{output_csv} 저장 완료! (총 {len(df_output)}건)')