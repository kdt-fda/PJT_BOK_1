import re
import csv
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {"User-Agent": "Mozilla/5.0"}

# 0) 설정
SEARCHWORD = "금리"
STARTDATE = "2012.01.01"
ENDDATE   = "2025.12.31"
SORT      = "desc"
OUT_CSV   = "hani_news.csv"
MAX_WORKERS = 10

# 한겨레 기사 URL 패턴
ARTICLE_URL_RE = re.compile(r"^https?://(www\.)?hani\.co\.kr/arti/.+/\d+\.html$")

# 1) 검색 URL 만들기
def build_search_url(page: int) -> str:
    base = "https://search.hani.co.kr/search/newslist"
    params = {
        "searchword": SEARCHWORD,
        "sort": SORT,
        "startdate": STARTDATE,
        "enddate": ENDDATE,
        "dt": "searchPeriod",
        "page": page,
    }
    return f"{base}?{urlencode(params)}"

# 2) 검색결과 페이지에서 기사 URL만 뽑기
def extract_article_urls_from_search(html: str):
    soup = BeautifulSoup(html, "html.parser")
    urls = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.startswith("//"):
            href = "https:" + href
        if ARTICLE_URL_RE.match(href):
            urls.add(href)

    return sorted(urls)

# 3) 기사 페이지에서 날짜/제목/본문 파싱
def parse_article(url: str):
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # --- 제목 ---
        title = ""
        title_el = soup.select_one('h3[class*="ArticleDetailView_title"], h1')
        if title_el:
            title = title_el.get_text(strip=True)
        if not title:
            title = "제목없음"

        # --- 날짜 ---
        date = "날짜없음"

        # 1) meta published_time 우선 시도
        meta = soup.select_one('meta[property="article:published_time"], meta[property="og:article:published_time"]')
        if meta and meta.get("content"):
            m = re.search(r"\d{4}-\d{2}-\d{2}", meta["content"])
            if m:
                date = m.group(0)

        # 2) 등록/수정 텍스트에서 날짜 찾기
        if date == "날짜없음":
            # 페이지 구조상 date list에 "등록"이 들어있음
            txt = soup.get_text(" ", strip=True)
            m = re.search(r"등록\s*(\d{4}-\d{2}-\d{2})", txt)
            if m:
                date = m.group(1)
            else:
                m2 = re.search(r"(\d{4}-\d{2}-\d{2})\s*\d{2}:\d{2}", txt)
                if m2:
                    date = m2.group(1)

        # --- 본문 ---
        paragraphs = []
        for p in soup.select('div.article-text p.text, div[class*="article-text"] p.text, p.text'):
            t = p.get_text(" ", strip=True)
            if not t:
                continue
            # 잡음 제거
            if t in ("광고", "기사을 읽어드립니다", "기사를 읽어드립니다"):
                continue
            paragraphs.append(t)

        content = "\n".join(paragraphs).strip()

        # 본문이 비었으면 fallback
        if not content:
            body_text = soup.get_text("\n", strip=True)
            body_text = re.sub(r"\n{3,}", "\n\n", body_text)
            content = body_text[:5000]

        full_text = f"{title}\n\n{content}".strip()

        return [date, full_text]

    except Exception as e:
        return None

# 4) 전체 실행
def main():
    all_urls = []
    page = 1
    consecutive_no_new = 0

    MAX_RETRY = 3
    RETRY_BASE_SLEEP = 1.0
    STOP_NO_NEW_PAGES = 5 

    print(">>> 1단계: 검색결과 페이지에서 URL 수집 시작")

    while True:
        search_url = build_search_url(page)

        # (A) 검색 페이지 요청: 실패 시 재시도
        html = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                res = requests.get(search_url, headers=HEADERS, timeout=15)
                res.raise_for_status()
                html = res.text
                break
            except Exception as e:
                wait = RETRY_BASE_SLEEP * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(f"❌ 검색 페이지 오류(page={page}) attempt={attempt}/{MAX_RETRY}: {e}")
                if attempt < MAX_RETRY:
                    print(f"   ↳ {wait:.2f}s 후 재시도...")
                    time.sleep(wait)
                else:
                    print(f"   ↳ 재시도 실패. page={page}는 건너뛰고 다음 페이지로 진행합니다.")
                    html = None

        if html is None:
            page += 1
            time.sleep(random.uniform(0.5, 1.2))
            continue

        # (B) URL 추출
        urls = extract_article_urls_from_search(html)

        # (디버깅용)
        if page == 1 and len(urls) == 0:
            print("⚠️ [page=1] 기사 URL이 0개입니다. (셀렉터/파라미터 확인 필요)")
            print("---- 응답 HTML 앞부분(800자) ----")
            print(html[:800])
            print("--------------------------------")
            return

        # (C) 중복 제거하며 누적
        new_cnt = 0
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
                new_cnt += 1

        print(f"[page={page}] urls={len(urls)} / new={new_cnt} / total={len(all_urls)}")

        # (D) 종료 조건
        if len(urls) == 0:
            print(">>> 검색결과가 더 이상 없어 종료합니다.")
            break

        if new_cnt == 0:
            consecutive_no_new += 1
            
            if consecutive_no_new >= STOP_NO_NEW_PAGES:
                print(f"⚠️ new=0이 {STOP_NO_NEW_PAGES}페이지 연속이라 종료합니다.")
                break
        else:
            consecutive_no_new = 0

        page += 1
        time.sleep(random.uniform(0.3, 0.8))

    print(f"\n>>> 총 URL: {len(all_urls)}개")
    if len(all_urls) == 0:
        print("URL이 0개라 종료합니다. (셀렉터/파라미터 확인 필요)")
        return

    print(f"\n>>> 2단계: 기사 본문 수집 시작 (Thread={MAX_WORKERS})")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(parse_article, url): url for url in all_urls}

        for i, fut in enumerate(as_completed(future_map), start=1):
            r = fut.result()
            if r:
                results.append(r)

            if i % 50 == 0:
                print(f"[{i}/{len(all_urls)}] 수집 진행중... (성공 {len(results)})")

            time.sleep(random.uniform(0.01, 0.05))

    # 날짜순 정렬
    results.sort(key=lambda x: x[0], reverse=True)

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "full_text"])
        w.writerows(results)

    print(f"\n✅ 저장 완료: {OUT_CSV} (rows={len(results)})")


if __name__ == "__main__":
    main()
