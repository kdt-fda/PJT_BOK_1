import pandas as pd
import kss
import gc
from ekonlpy.sentiment import MPCK
from tqdm import tqdm

def initialize_kss():
    print('문장 분리 엔진 설정 중...')
    try:
        kss.split_sentences('테스트', backend='mecab')
        print('Mecab 엔진 로드 완료')
        return 'mecab'
    except Exception as e:
        print(f'Mecab 로드 실패({e}), 기본 엔진으로 전환')
        return 'keunmago'

def preprocess_news(input_file, output_file, batch_size=2000):
    # 엔진 및 분석기 초기화
    backend = initialize_kss()
    analyzer = MPCK()
    
    # 데이터 로드
    print(f'데이터 읽는 중: {input_file}')
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    raw_data = list(zip(df['date'], df['cleansed_text']))
    
    final_data = []
    print(f'분석 시작, 총 {len(raw_data)}건')
    
    # 루프 실행
    for i, (date, content) in enumerate(tqdm(raw_data, desc=f"Processing {input_file}")):
        if pd.isnull(content) or not str(content).strip():
            continue
        
        try:
            # 문장 분리
            sentences = kss.split_sentences(str(content), backend=backend)
            
            for sent in sentences:
                if len(sent) < 5: continue # 너무 짧은 문장 제외
                
                # 토큰화 및 품사 필터링 (명사, 동사, 형용사, 부사, 부정어)
                tokens = analyzer.tokenize(sent)
                filtered = [t for t in tokens if '/' in t and 
                            t.split('/')[-1].startswith(('N', 'V', 'M'))]
                
                if filtered:
                    final_data.append([str(date), ",".join(filtered)])
            
            # 중간 저장
            if (i + 1) % batch_size == 0:
                save_header = True if (i + 1) == batch_size else False
                save_mode = 'w' if (i + 1) == batch_size else 'a'
                
                temp_df = pd.DataFrame(final_data, columns=['date', 'tokens'])
                temp_df.to_csv(output_file, index=False, encoding='utf-8-sig', 
                               mode=save_mode, header=save_header)
                
                final_data = []
                gc.collect()

        except Exception as e:
            # 에러 발생 시 건너뛰기
            continue

    # 잔여 데이터 저장
    if final_data:
        temp_df = pd.DataFrame(final_data, columns=['date', 'tokens'])
        has_header = not pd.io.common.file_exists(output_file)
        temp_df.to_csv(output_file, index=False, encoding='utf-8-sig', 
                       mode='a', header=has_header)

    print(f"\n🎉 {output_file} 저장 완료!")

# 실제 실행
if __name__ == "__main__":
    target_files = [
        ('chosun_news_cleansing.csv', 'chosun_news_tokenize.csv'),
        ('donga_news_cleansing.csv', 'donga_news_tokenize.csv'),
        ('khan_news_cleansing.csv', 'khan_news_tokenize.csv')
    ]
    
    for input, output in target_files:
        preprocess_news(input, output)