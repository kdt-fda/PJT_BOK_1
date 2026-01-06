import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str): return ''
    
    # 줄바꿈 및 이메일/URL 제거
    text = re.sub(r'\\n|\n|\r', ' ', text)
    text = re.sub(r'[a-zA-Z0-9\.\-\_+]+\@[a-zA-Z0-9\.\-\_\+]+\.[a-zA-Z]{2,}', ' ', text)
    text = re.sub(r'http[s]?://\S+', ' ', text)

    # 기자명 제거
    text = re.sub(r'[가-힣]{2,4}\s*기자', ' ', text)

    # 특수기호 제거 (문장부호 .,?! 퍼센트 기호 % 제외)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!%]', ' ', text)

    # 공백 및 마침표 정리
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_cleansing(input_path, output_path, min_length=200, keyword='금리'):
    print(f'{input_path} 처리 시작...')
    
    # 데이터 로드
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f'파일을 찾을 수 없습니다: {input_path}')
        return

    # 텍스트 전처리
    df['cleansed_text'] = df['full_text'].apply(clean_text)

    # 필터링 조건 설정
    # 1) 키워드 포함 여부
    keyword_condition = df['full_text'].str.contains(keyword, na=False)
    # 2) 공백 제외 글자 수 조건
    length_condition = df['cleansed_text'].apply(lambda x: len(str(x).replace(" ", "")) >= min_length)

    # 필터링 적용 및 복사
    df_filtered = df[keyword_condition & length_condition].copy()

    # 불필요한 컬럼 삭제
    if 'full_text' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['full_text'])

    # 결과 통계 출력
    before_count = len(df)
    after_count = len(df_filtered)
    
    print(f'전처리 완료: {input_path}')
    print(f'   - 원본 기사 수: {before_count}건')
    print(f'   - 필터링 후(키워드 미포함 및 {min_length}자 미만 제거): {after_count}건')
    print(f'   - 제거된 기사 수: {before_count - after_count}건')

    # 파일 저장
    df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 결과 저장 완료: {output_path}")

# 실제 실행부
if __name__ == "__main__":
    tasks = [
        ('chosun_news.csv', 'chosun_news_filtered.csv'),
        ('donga_news.csv', 'donga_news_filtered.csv'),
        ('khan_news.csv', 'khan_news_filtered.csv')
    ]

    for input_file, output_file in tasks:
        run_cleansing(input_file, output_file)