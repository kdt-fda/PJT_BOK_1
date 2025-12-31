import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime

#  저장 경로 설정
save_dir = "./crawling_minutes.pdf"
os.makedirs(save_dir, exist_ok=True)
listcont_url = "https://www.bok.or.kr/portal/singl/newsData/listCont.do"

params = {
    "pageIndex": 1,
    "targetDepth": 3,
    "menuNo": 201154,
    "depth2": 200038,
    "depth3": 201154,
    "depth4": 200789,
    "searchCnd": 1,
    "searchKwd": "",
    "sort": 1,
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.bok.or.kr/portal/singl/newsData/list.do"
}

res = requests.get(listcont_url, params=params, headers=headers)
soup = BeautifulSoup(res.text, "html.parser")

rows = soup.select("li.bbsRowCls")
print("게시글 수:", len(rows))   

res.status_code
len(res.text)
from datetime import datetime

START_DATE = datetime(2012, 1, 1)
END_DATE = datetime(2025, 12, 31)

def download_pdf(pdf_url, save_path):
    try:
        r = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"},timeout = 15)
        r.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(r.content)

        print(f"   다운로드 완료: {save_path}")

    except Exception as e:
        print(f"   다운로드 실패: {pdf_url}")
        print(f"     에러: {e}")

# 페이지 자동 순회
page = 1
while True:
    params["pageIndex"] = page
    res = requests.get(listcont_url, params=params, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    rows = soup.select("li.bbsRowCls")

    if not rows:
        break

    stop = False
    for row in rows:
        title = row.select_one("a.title").get_text(strip=True)
        date_text = row.select_one("span.date").get_text(strip=True)
        date_str = date_text.replace("등록일", "").strip()
        date_obj = datetime.strptime(date_str, "%Y.%m.%d")
        

        if date_obj < START_DATE:
            stop = True
            break

        if date_obj > END_DATE:
            continue

        print(date_str, title)

        detail_url = row.select_one("a.title")["href"]

        if not detail_url.startswith("http"):
            detail_url = "https://www.bok.or.kr" + detail_url

        detail_res = requests.get(detail_url, headers=headers)
        detail_soup = BeautifulSoup(detail_res.text, "html.parser")

        pdf_link = detail_soup.select_one("a[href$='.pdf']")
        if not pdf_link:
            print("  ⚠ PDF 없음")
            continue

        pdf_url = pdf_link["href"]
        if not pdf_url.startswith("http"):
            pdf_url = "https://www.bok.or.kr" + pdf_url

        print("  PDF:", pdf_url)

        file_name = f"{date_str}_의사록.pdf"
        save_path = os.path.join(save_dir, file_name)

        if os.path.exists(save_path):
            continue
        
        download_pdf(pdf_url, save_path)


    if stop:
        break

    page += 1

import pdfplumber
import os
import re
import pandas as pd

PDF_DIR = "./crawling_minutes.pdf"
OUTPUT_FILE = "../preprocessing/minutes_text.txt"


#with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
#        pass

for filename in sorted(os.listdir(PDF_DIR)):
    if not filename.endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_DIR, filename)

    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

    full_text = "\n".join(full_text)

    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        f_out.write(f"---{filename}---\n")
        f_out.write(full_text + "\n\n")
