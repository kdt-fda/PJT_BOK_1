import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return ''

    # 줄바꿈 기호 제거
    text = re.sub(r'\\n', ' ', text)

    # 이메일 제거
    text = re.sub(r'[a-zA-Z0-9\.\-\_+]+\@[a-zA-Z0-9\.\-\_\+]+\.[a-zA-Z]{2,}', ' ', text)

    # URL 제거
    text = re.sub(r'http[s]?://\S+', ' ', text)

    # 숫자 제거
    text = re.sub(r'\d+(\.\d+)?', ' ', text)

    # 괄호, 따옴표, 슬래시, 쉼표, 문장부호 제거
    text = re.sub(r'[()\[\]{}<>\"\'\\\,\.\`]', ' ', text)

    # 한글 제외 문자 제거
    text = re.sub(r'[^가-힣\s]', ' ', text)

    # 기자명 제거
    text = re.sub(r'[가-힣]{2,4}\s*기자', ' ', text)

    # 불용어 제거
    stopwords = ['무단전재', '재배포', '금지', '저작권', '지난', '최근', '대해', '위해',
                 '통해', '이번', '요즘', '일보', '신문', '부문', '기사', '올해', '내년',
                 '등의', '이래', '오늘', '어제', '내일', '그간', '아무런', '작년']
    
    for word in stopwords:
        text = text.replace(word, ' ')

    # 공백 한 칸으로 정리
    text = re.sub(r'\s+', ' ', text)

    return text

# 조선일보 실행
df = pd.read_csv('chosun_news.csv')

print(f'총 {len(df)}건의 데이터 전처리 시작...')

df['cleansed_text'] = [clean_text(text) for text in df['full_text']]

# 금리 포함 여부 확인
keyword_condition = df['full_text'].str.contains('금리', na=False)

# 200자 미만 필터링
length_condition = df['cleansed_text'].apply(lambda x: len(str(x).replace(" ", "")) >= 200)

df_filtered = df[keyword_condition & length_condition].copy()

# 원본 본문 삭제
df_filtered = df_filtered.drop(columns=['full_text'])

# 결과 확인
print('')
print(f'전처리 완료')

before_count = len(df)
after_count = len(df_filtered)

print(f'원본 기사 수: {before_count}건')
print(f'제거된 기사(200자 미만): {before_count - after_count}건')
print(f'최종 남은 기사 수: {after_count}건')

# 최종 파일 저장
output_path = 'cleansed_chosun_news.csv'
df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')

# 동아일보 실행
df = pd.read_csv('donga_news.csv')

print(f'총 {len(df)}건의 데이터 전처리 시작...')

df['cleansed_text'] = [clean_text(text) for text in df['full_text']]

# 금리 포함 여부 확인
keyword_condition = df['full_text'].str.contains('금리', na=False)

# 200자 미만 필터링
length_condition = df['cleansed_text'].apply(lambda x: len(str(x).replace(" ", "")) >= 200)

df_filtered = df[keyword_condition & length_condition].copy()

# 원본 본문 삭제
df_filtered = df_filtered.drop(columns=['full_text'])

# 결과 확인
print('')
print(f'전처리 완료')

before_count = len(df)
after_count = len(df_filtered)

print(f'원본 기사 수: {before_count}건')
print(f'제거된 기사(200자 미만): {before_count - after_count}건')
print(f'최종 남은 기사 수: {after_count}건')

# 최종 파일 저장
output_path = 'cleansed_donga_news.csv'
df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')

# 경향신문 실행
df = pd.read_csv('khan_news.csv')

print(f'총 {len(df)}건의 데이터 전처리 시작...')

df['cleansed_text'] = [clean_text(text) for text in df['full_text']]

# 금리 포함 여부 확인
keyword_condition = df['full_text'].str.contains('금리', na=False)

# 200자 미만 필터링
length_condition = df['cleansed_text'].apply(lambda x: len(str(x).replace(" ", "")) >= 200)

df_filtered = df[keyword_condition & length_condition].copy()

# 원본 본문 삭제
df_filtered = df_filtered.drop(columns=['full_text'])

# 결과 확인
print('')
print(f'전처리 완료')

before_count = len(df)
after_count = len(df_filtered)

print(f'원본 기사 수: {before_count}건')
print(f'제거된 기사(200자 미만): {before_count - after_count}건')
print(f'최종 남은 기사 수: {after_count}건')

# 최종 파일 저장
output_path = 'cleansed_khan_news.csv'
df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')