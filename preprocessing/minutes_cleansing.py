import re
import os
import pandas as pd
import glob

# 공통 전처리
def clean_line(line: str) -> str:
    line = line.strip()
    # PDF 페이지 번호 제거: - 1 -
    if re.match(r"^-\s*\d+\s*-\s*$", line):
        return ""
    return line

# 숫자/특수기호 제거
def remove_numbers_symbols(text: str) -> str:
    if text is None:
        return text

    text = re.sub(r"\d+(\.\d+)?", " ", text)

    text = re.sub(r"[%‰]", " ", text)

    text = re.sub(r"[()\[\]{}<>\"']", " ", text)

    text = re.sub(r"[.,;:!?…]", " ", text)

    text = re.sub(r"[^가-힣a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# 회의일자 추출 (텍스트 내부)
def parse_meeting_date(full_text: str):
    m = re.search(
        r"1\.\s*일\s*(?:자|시)\s*(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일",
        full_text
    )
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    return f"{y:04d}-{mth:02d}-{d:02d}"

# 공개일자 추출 (파일명 기준)
def parse_release_date_from_header(header_line: str):
    # ---2024.12.24_의사록.pdf---  -> 2024-12-24
    m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", header_line)
    if not m:
        return None
    y, mth, d = m.groups()
    return f"{y}-{mth}-{d}"

def remove_section_titles(text: str) -> str:
    if text is None:
        return ""

    text = re.sub(r"(?:\(\s*\d+\s*\)\s*)?위원\s*토\s*의\s*내\s*용", " ", text)

    text = re.sub(r"(?:\(\s*\d+\s*\)\s*)?토\s*의\s*내\s*용", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# 가볍게 의사록 분리
def split_light_minutes(full_text: str):
    # (별첨) 기준 분리
    parts = re.split(r"\(\s*별\s*첨\s*\)", full_text, maxsplit=1)
    main_text = parts[0]
    appendix = parts[1].strip() if len(parts) == 2 else ""

    # 토의내용 기준 분리
    discussion_re = re.compile(
        r"(?:\(\s*\d+\s*\)\s*)?"      # (3) 같은 번호가 있어도/없어도
        r"위원\s*토\s*의\s*내\s*용"   
    )
    m_disc = discussion_re.search(main_text)
    if m_disc:
        meta = main_text[:m_disc.start()].strip()
        body_all = main_text[m_disc.start():].strip()
    else:
        return main_text, "", "", appendix

    # DECISION 판별 패턴 (토의결론 포함)
    decision_re = re.compile(
        r"<\s*(의안|보고)\s*제|"
        r"심\s*의\s*결\s*과|"
        r"토\s*의\s*결\s*론|"
        r"의\s*결\s*사\s*항|"
        r"원\s*안\s*대\s*로\s*가\s*결|"
        r"<\s*붙\s*임\s*>"
    )
    m_dec = decision_re.search(body_all)
    if m_dec:
        body = body_all[:m_dec.start()].strip()
        decision = body_all[m_dec.start():].strip()
    else:
        # decision 섹션 못 찾으면 전부 body로
        body = body_all.strip()
        decision = ""

    body = remove_section_titles(body)
    decision = remove_section_titles(decision)


    return meta, body, decision, appendix

# txt 1개 → CSV row 생성
def raw_minutes_to_rows(raw_txt_path: str):
    with open(raw_txt_path, "r", encoding="utf-8") as f:
        lines = [clean_line(l) for l in f.readlines()]

    # 회차 헤더: ---2024.12.24_의사록.pdf---
    header_re = re.compile(r"^---\s*\d{4}\.\d{2}\.\d{2}_.+?---$")

    docs = []
    current_header = None
    buf = []

    for line in lines:
        if not line:
            continue

        if header_re.match(line):
            # 이전 회차 저장
            if current_header is not None:
                docs.append((current_header, " ".join(buf).strip()))
            # 새 회차 시작
            current_header = line
            buf = []
        else:
            buf.append(line)

    # 마지막 회차 저장
    if current_header is not None:
        docs.append((current_header, " ".join(buf).strip()))

    all_rows = []

    for header, text in docs:
        release_date = parse_release_date_from_header(header)
        meeting_date = parse_meeting_date(text)

        meta, body, decision, appendix = split_light_minutes(text)

        def add_row(text_type, t):
            if t and t.strip():
                all_rows.append({
                    "meeting_date": meeting_date,
                    "release_date": release_date,
                    "text_date": release_date,      # 라벨링 기준일
                    "text_type": text_type,         # META/BODY/DECISION/APPENDIX
                    "text": t.strip(),
                    "source_type": "minutes"
                })

        add_row("META", meta)
        add_row("BODY", body)
        add_row("DECISION", decision)
        add_row("APPENDIX", appendix)

    return all_rows

# 실행부: 폴더 내 모든 txt 처리
all_rows = raw_minutes_to_rows("./minutes_text.txt")

df = pd.DataFrame(all_rows)

df["text"] = df["text"].astype(str).apply(remove_numbers_symbols)

df.to_csv("minutes_stage1.csv", index=False, encoding="utf-8-sig")

# sanity check
print(df["text_type"].value_counts())
print(df[["meeting_date", "release_date"]].drop_duplicates())