import re
import json
import pandas as pd

RAW_TXT_PATH = "./minutes_text.txt"


def remove_numbers_symbols(text: str) -> str:
    if text is None:
        return ""

    # 숫자 제거 (정수, 소수)
    text = re.sub(r"\d+(\.\d+)?", " ", text)

    # 퍼센트/퍼밀
    text = re.sub(r"[%‰]", " ", text)

    # 괄호/따옴표
    text = re.sub(r"[()\[\]{}<>\"']", " ", text)

    # 문장부호
    text = re.sub(r"[.,;:!?…]", " ", text)

    # 한글/영문/공백만 남김
    text = re.sub(r"[^가-힣a-zA-Z\s]", " ", text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_release_date_from_header(header_line: str):
    m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", header_line)
    if not m:
        return None
    y, mth, d = m.groups()
    return f"{y}-{mth}-{d}"


def parse_meeting_date(full_text: str):
    # full_text는 문장 합친 문자열(날짜 파싱용)
    m = re.search(
        r"1\.\s*일\s*(?:자|시)\s*(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일",
        full_text
    )
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    return f"{y:04d}-{mth:02d}-{d:02d}"


def remove_section_titles_sent(sent: str) -> str:
    if not sent:
        return ""
    sent = re.sub(r"(?:\(\s*\d+\s*\)\s*)?위원\s*토\s*의\s*내\s*용", " ", sent)
    sent = re.sub(r"(?:\(\s*\d+\s*\)\s*)?토\s*의\s*내\s*용", " ", sent)
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent


# 문장 리스트 기반 섹션 분리
#    - "별첨" 이후는 APPENDIX
#    - "위원 토의내용" 이후 BODY 시작
#    - decision trigger 이후 DECISION

DISCUSSION_RE = re.compile(r"(?:\(\s*\d+\s*\)\s*)?위원\s*토\s*의\s*내\s*용")
APPENDIX_RE  = re.compile(r"\(\s*별\s*첨\s*\)")

DECISION_RE = re.compile(
    r"<\s*(의안|보고)\s*제|"
    r"심\s*의\s*결\s*과|"
    r"토\s*의\s*결\s*론|"
    r"의\s*결\s*사\s*항|"
    r"원\s*안\s*대\s*로\s*가\s*결|"
    r"<\s*붙\s*임\s*>"
)

def split_minutes_by_sentences(sentences):
    """
    sentences: 한 회차(한 pdf)에서 추출된 '문장 줄' 리스트
    return: dict {META:[...], BODY:[...], DECISION:[...], APPENDIX:[...]}
    """
    sections = {"META": [], "BODY": [], "DECISION": [], "APPENDIX": []}

    state = "META"
    in_appendix = False

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # (별첨) 만나면 이후는 APPENDIX
        if APPENDIX_RE.search(s):
            in_appendix = True
            state = "APPENDIX"
            continue

        if in_appendix:
            sections["APPENDIX"].append(s)
            continue

        # 토의내용 시작 전까지 META
        if state == "META" and DISCUSSION_RE.search(s):
            state = "BODY"
            # "위원 토의내용" 타이틀 문장 제거
            continue

        # BODY 중 DECISION 시작 트리거
        if state == "BODY" and DECISION_RE.search(s):
            state = "DECISION"

        sections[state].append(s)

    # 섹션 타이틀 제거(문장단위)
    for k in ["META", "BODY", "DECISION", "APPENDIX"]:
        sections[k] = [remove_section_titles_sent(x) for x in sections[k]]
        sections[k] = [x for x in sections[k] if x]  # 빈문장 제거

    return sections


HEADER_RE = re.compile(r"^---\s*\d{4}\.\d{2}\.\d{2}_.+?---$")

def load_docs_from_minutes_txt(path: str):
    """
    return: list of dict
      [{
        header: str,
        release_date: str,
        sentences: [str, ...]
      }, ...]
    """
    docs = []
    current_header = None
    buf = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if HEADER_RE.match(line):
                if current_header is not None:
                    docs.append({
                        "header": current_header,
                        "release_date": parse_release_date_from_header(current_header),
                        "sentences": buf
                    })
                current_header = line
                buf = []
            else:
                buf.append(line)

    if current_header is not None:
        docs.append({
            "header": current_header,
            "release_date": parse_release_date_from_header(current_header),
            "sentences": buf
        })

    return docs


docs = load_docs_from_minutes_txt(RAW_TXT_PATH)

rows_list_json = []   # 팀규칙 형태를 유지(문장 리스트를 json 문자열로)
rows_exploded = []    

for doc in docs:
    release_date = doc["release_date"]
    meeting_date = parse_meeting_date(" ".join(doc["sentences"]))

    sections = split_minutes_by_sentences(doc["sentences"])

    # 섹션별로 문장 클렌징(문장단위)
    for sec, sents in sections.items():
        cleaned_sents = [remove_numbers_symbols(s) for s in sents]
        cleaned_sents = [s for s in cleaned_sents if s]  # 비어있으면 제거

        # 팀규칙: [date, [sent1, sent2...]]
        rows_list_json.append({
            "section": sec,
            "release_date": release_date,
            "sentences": json.dumps(cleaned_sents, ensure_ascii=False),
            "source_type": "minutes"
        })


df_list = pd.DataFrame(rows_list_json)

df_list.to_csv("minutes_stage1_list.csv", index=False, encoding="utf-8-sig")

print(df_list["section"].value_counts())



df = pd.read_csv("minutes_stage1_list.csv")

# BODY만 필터링
df_body = df[df["section"] == "BODY"].copy()

df_body = df_body[["release_date", "sentences"]]

# BODY가 비어있는 행 제거
def has_sentences(x):
    try:
        sents = json.loads(x)
        return len(sents) > 0
    except:
        return False

df_body = df_body[df_body["sentences"].apply(has_sentences)]

# BODY 전용 CSV 저장
df_body.to_csv(
    "minutes_filtered.csv",
    index=False,
    encoding="utf-8-sig"
)