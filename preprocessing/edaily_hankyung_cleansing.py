import pandas as pd
import re

def clean_text(text):
    if not text:
        return ""
    
    # 기자명 및 이메일 제거
    # 이름(2~4자) + 기자 + 이메일 형태 또는 단순히 기자명만 있는 경우 처리
    text = re.sub(r'[가-힣]{2,4}\s?기자\s?\(?[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\)?', '', text) # 이름 기자(이메일)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text) # 남은 이메일
    text = re.sub(r'[가-힣]{2,4}\s?기자', '', text) # OOO 기자 혹은 OOO기자 삭제
    
    # 대괄호 [] 및 그 안의 내용 제거
    text = re.sub(r'\[.*?\]', '', text)
    
    # 저작권 관련 상용구 및 언론사 패턴
    patterns = [
        r'\(?ⓒ\s?.*\)?',
        r'저작권자\s?.*무단\s?전재\s?재배포\s?금지',
        r'재배포\s?금지',
        r'무단\s?전재\s?및\s?재배포\s?금지',
        r'한국경제\s?.*기자', 
        r'구독신청\s?.*확인',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    # 특수기호 제거 (분석에 필요한 문장부호 .,?! 만 남김)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!%]', ' ', text)
    
    # 불필요한 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def length_filtering(file_name, output_name, min_length):
    try:
        # 열이름을 date, full_text로 해서 파일 열기 
        df = pd.read_csv(file_name, header=None, names=['date', 'full_text'], encoding="utf-8-sig")
        initial_count = len(df)
        
        # 빈 기사, 공백 전처리 및 문자열 형태로 변환
        df['full_text'] = df['full_text'].fillna('').astype(str)
        
        # 뉴스 내용 클렌징
        print(f"[{file_name}] 노이즈 제거 중...")
        df['full_text'] = df['full_text'].apply(clean_text)
        
        # min_length 미만 필터링
        df['full_text'] = df['full_text'].str.strip()
        
        # 길이 필터링 수행
        len_condition = df['full_text'].str.len() >= min_length
        # '금리' 키워드 포함 필터링 수행
        keyword_condition = df['full_text'].str.contains('금리', na=False)
        
        # 최종 필터링 적용
        df_filtered = df[len_condition & keyword_condition].copy()
        
        final_count = len(df_filtered)
        
        # 통계 계산
        short_deleted = initial_count - len(df[len_condition])
        keyword_deleted = len(df[len_condition]) - final_count
        
        df_filtered.to_csv(output_name, index=False, header=False, encoding="utf-8-sig")
        
        print(f"- 원본 기사: {initial_count}건")
        print(f"- 길이 미달 삭제: {short_deleted}건")
        print(f"- '금리' 미포함 삭제: {keyword_deleted}건")
        print(f"- 총 삭제 기사: {initial_count - final_count}건")
        print(f"- 최종 기사: {final_count}건, {output_name} 저장 완료\n")
        
        return df_filtered

    except FileNotFoundError:
        print(f"오류: {file_name} 파일을 찾을 수 없습니다.\n")
        return None

df_edaily = length_filtering('edaily_news_all_crawl.csv', 'edaily_news_all_cleansing.csv', 150)
df_hankyung = length_filtering('hankyung_news_all_crawl.csv', 'hankyung_news_all_cleansing.csv', 150)