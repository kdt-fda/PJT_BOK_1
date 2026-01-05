import pandas as pd
import re
import kss
from ekonlpy.sentiment import MPCK
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 전역 변수 설정
mpck_instance = None

def init_worker():
    global mpck_instance
    if mpck_instance is None:
        mpck_instance = MPCK()

# 문장 분리
def get_sentences(row):
    date = str(row[0]) 
    content = row[1]
    
    # 빈 값이나 공백만 있는 데이터 처리
    if pd.isnull(content) or not str(content).strip():
        return [str(date), []]
    
    global mpck_instance
    try:
        text = re.sub(r'\.{2,}', '. ', str(content))
        sentences = kss.split_sentences(text)
        return [str(date), sentences]
    except:
        return [str(date), []]

# 분리된 문장 리스트를 받아 형태소 분석 수행
def get_pos_list(sent_row):
    date, sentences = sent_row
    global mpck_instance
    
    pos_analyzed = []
    for sent in sentences:
        # eKoNLPy의 MPCK를 이용한 토큰화 (문장별)
        tokens = mpck_instance.tokenize(sent)
        pos_analyzed.append(tokens)
        
    return [date, pos_analyzed]

def run_analysis(file_path):
    df = pd.read_csv(file_path, header=None, names=['date', 'full_text'])
    raw_data = [[str(d), str(t)] for d, t in zip(df['date'], df['full_text'])]
    
    num_cores = 4 # 멀티 프로세스 코어
    
    # 문장 분리
    print(f"\n[1/2] 문장 분리 시작: {file_path}")
    with Pool(processes=num_cores, initializer=init_worker) as pool:
        sent_split_results = list(tqdm(pool.imap(get_sentences, raw_data, chunksize=20), 
                                    total=len(raw_data)))
    
    # 싱글 코어 코드
    # sent_split_results = []
    # for row in tqdm(raw_data):
    #     result = get_sentences(row)
    #     sent_split_results.append(result)
    # print(sent_split_results)
    

    # 형태소 분석
    print(f"\n[2/2] 형태소 분석 시작")
    with Pool(processes=num_cores, initializer=init_worker) as pool:
        final_pos_results = list(tqdm(pool.imap(get_pos_list, sent_split_results, chunksize=20), 
                                    total=len(sent_split_results)))
    
    # 싱글 코어 코드
    # final_pos_results = []
    # for sent_row in tqdm(sent_split_results):
    #     result = get_pos_list(sent_row) 
    #     final_pos_results.append(result)
    # print(final_pos_results)
     
    return sent_split_results, final_pos_results

if __name__ == "__main__":
    # init_worker() # 싱글 코어 시 주석 해제
    
    # 분석 실행
    edaily_sent, edaily_pos = run_analysis('edaily_news_all_cleansing.csv')
    hankyung_sent, hankyung_pos = run_analysis('hankyung_news_all_cleansing.csv')
    
    # 최종 결과 CSV 저장
    output_df1 = pd.DataFrame(edaily_pos, columns=['date', 'pos_tokens'])
    output_df1.to_csv('edaily_news_all_tokenize.csv', index=False, encoding='utf-8-sig')
    
    output_df2 = pd.DataFrame(hankyung_pos, columns=['date', 'pos_tokens'])
    output_df2.to_csv('hankyung_news_all_tokenize.csv', index=False, encoding='utf-8-sig')
    
    print("\nCSV 파일 생성이 완료되었습니다.")