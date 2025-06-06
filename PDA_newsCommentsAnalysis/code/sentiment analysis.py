# ==============================================================================
# 라이브러리 임포트 및 한글 폰트 설정
# ==============================================================================

# print("\n--- 라이브러리 임포트 및 한글 폰트 설정 중 ---")
import re
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt # 폰트 설정용으로만 사용 (시각화 제외)
import seaborn as sns           # 폰트 설정용으로만 사용 (시각화 제외)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm.notebook import tqdm
from collections import Counter
# from wordcloud import WordCloud # 워드클라우드 생성을 원치 않으면 이 줄과 관련 코드 삭제
# from konlpy.tag import Okt      # 워드클라우드 생성을 원치 않으면 이 줄과 관련 코드 삭제
import matplotlib.font_manager as fm

# try:
#     font_path = 'C:/Windows/Fonts/NanumGothic.ttf' # 당신의 폰트 경로로 변경!
#     # 이 부분은 'C:/Windows/Fonts/malgun.ttf' 가 일반적입니다.
#     # 만약 바탕 화면 폴더 안에 malgun.ttf 파일을 직접 넣었다면 이 경로가 맞을 수 있습니다.
#     fm.fontManager.addfont(font_path)
#     plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
#     plt.rcParams['axes.unicode_minus'] = False
#     print(f"한글 폰트 '{plt.rcParams['font.family']}' 설정 완료: {font_path}")
# except Exception as e:
#     print(f"한글 폰트 설정에 실패했습니다. 기본 폰트로 시각화를 진행합니다. 오류: {e}")
# print("-" * 50)


# ==============================================================================
# 감성 분석 모델 로드 및 함수 정의 (KoTE 모델 활용 - 43가지 감정 처리)
# ==============================================================================

print("\n--- 감성 분석 모델 로드 및 함수 정의 중 ---")

# 감성 분석에 사용할 디바이스 설정: GPU가 있다면 GPU, 없으면 CPU
device = 0 if torch.cuda.is_available() else -1
print(f"감성 분석에 사용할 디바이스: {'GPU' if device == 0 else 'CPU'}")

sentiment_pipeline = None # 모델 로드 실패 시 None으로 유지

try:
    # hugging face에서 토크나이저와 모델 불러오기
    tokenizer = AutoTokenizer.from_pretrained("searle-j/kote_for_easygoing_people")
    model = AutoModelForSequenceClassification.from_pretrained("searle-j/kote_for_easygoing_people")

    # 모델의 실제 id2label 매핑을 동적으로 가져옵니다.
    # 이것이 43가지 감성 라벨을 정확하게 포함해야 합니다.
    global id_to_label
    id_to_label = model.config.id2label

    # id_to_label을 key (int) 기준으로 정렬하여, 나중에 결과 출력 시 순서를 맞춥니다.
    # 예: {0: '기쁨', 1: '슬픔', ...}
    id_to_label = {k: v for k, v in sorted(id_to_label.items())}

    print(f"모델이 인식하는 감성 라벨 수: {len(id_to_label)}")
    print(f"모델의 감성 라벨 매핑 예시: {list(id_to_label.items())[:5]} ... {list(id_to_label.items())[-5:]}")


    # 감정 분석 모델 설정 (TextClassificationPipeline 사용)
    sentiment_pipeline = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,          # gpu number, -1 if cpu used
        return_all_scores=True, # 모든 감성 라벨의 점수를 반환
        function_to_apply='sigmoid' # KoTE 모델은 시그모이드 사용
    )
    print("감성 분석 모델 로드 완료.")
except Exception as e:
    print(f"감성 분석 모델 로드 중 오류 발생: {e}")
    print("transformers 라이브러리 설치 및 인터넷 연결 상태를 확인해주세요.")
    print(f"상세 오류: {e}") # 상세 오류 메시지 출력
    sentiment_pipeline = None # 모델 로드 실패 시 None으로 유지

def analyze_sentiment_with_pipeline(texts):
    """
    텍스트 리스트에 대해 감성 분석을 수행하고, 각 텍스트의 43가지 감성 점수를

    Pandas DataFrame 형태로 반환합니다.
    """
    if not texts:
        return pd.DataFrame(columns=list(id_to_label.values())) # 빈 DataFrame 반환
    if sentiment_pipeline is None:
        print("감성 분석 모델이 로드되지 않아 분석을 건너뜝니다.")
        return pd.DataFrame(columns=list(id_to_label.values()))
    # texts=texts[0:1]
    results = sentiment_pipeline(texts)
    
    # 각 텍스트별로 43가지 감성의 점수를 리스트로 저장 (컬럼 순서 고정)
    all_probabilities_list = []
    
    for res_list_for_one_text in results: # 하나의 텍스트에 대한 감성 분석 결과 (list of dicts)
        # 43가지 감성 라벨 순서에 맞게 점수를 저장할 임시 배열 초기화
        # id_to_label의 key (0부터 42까지) 순서대로 배열에 점수를 채웁니다.
        current_text_scores = [0.0] * len(id_to_label)
        
        for item in res_list_for_one_text: # 각 감성 라벨에 대한 score 딕셔너리
            # 'LABEL_0', 'LABEL_1' 형태의 라벨에서 숫자 인덱스를 추출
            try:
                # label_id = int(item['label'].replace('label', ''))
                key = next((k for k, v in id_to_label.items() if v == item['label']), None)
                label_id = int(key)
                if 0 <= label_id < len(id_to_label): # 유효한 라벨 인덱스인지 확인
                    current_text_scores[label_id] = item['score']
            except ValueError:
                # 'LABEL_X' 형식이 아닌 다른 라벨이 있다면 경고
                print(f"경고: 예상치 못한 라벨 형식 발견: {item['label']}. 건너뜁니다.")
                continue
        all_probabilities_list.append(current_text_scores)
        
    # 결과가 데이터프레임이 되도록 변환
    # 컬럼 이름은 id_to_label의 값 (감성 이름)을 사용
    return pd.DataFrame(all_probabilities_list, columns=list(id_to_label.values()))

print("-" * 50)


# ==============================================================================
# 신문사별 월별 데이터 통합 및 감성 분석 메인 로직
# (상위 5개 감성 출력 포함)
# ==============================================================================

print("\n--- 신문사별 월별 데이터 통합 및 감성 분석 메인 로직 시작 ---")
this_dir= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir= os.path.join(this_dir, "src", "news_comments_filtered")

# base_dir = '../src/news_comments_filtered' # 당신의 실제 로컬 경로로 변경!

print(f"데이터를 읽어올 기본 경로: {base_dir}")


target_months = ['2024-12', '2025-01', '2025-02', '2025-03', '2025-04']
# target_months = ['2024-12']
all_results_by_month_and_org = {} # 월별, 신문사별, 감성별 평균 점수를 저장
all_comments_for_wordcloud = [] # 워드클라우드 생성을 위한 모든 댓글 통합 (옵션)

try:
    news_org_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    # news_org_dirs=[ '뉴시스', '동아일보', '매일경제', '세계일보', '연합뉴스', '오마이뉴스', '이데일리', '조선일보', '중앙일보', '한겨레', '한국경제']

except FileNotFoundError:
    print(f"\n오류: 기본 경로 '{base_dir}'를 찾을 수 없습니다. 로컬 파일 시스템 경로를 확인해주세요.")
    news_org_dirs = []

if not news_org_dirs:
    print(f"\n경고: '{base_dir}' 경로에서 신문사 폴더를 찾을 수 없습니다. 분석을 종료합니다.")
else:
    print(f"\n--- 다음 신문사 폴더에서 파일을 찾습니다: {news_org_dirs} ---")
for news_org in news_org_dirs:
    all_data_for_df1 = pd.DataFrame()
    for month in target_months:
        print(f"\n======== {month} 월 데이터 처리 중 ========")
        all_results_by_month_and_org[month] = {}
        count_a=0;

        count_a+=1
        print(f"    --- {month}월 {news_org} {count_a} 데이터 처리 중 ---")
        news_org_monthly_comments_list = []
        news_org_path = os.path.join(base_dir, news_org)
        
        daily_files = [f for f in os.listdir(news_org_path) if f.startswith(month) and f.endswith('.csv')]
        
        if not daily_files:
            print(f"      경고: {month}월 {news_org}에 해당하는 파일이 없습니다. 건너뜁니다.")
            continue

        for filename in daily_files:
            file_path = os.path.join(news_org_path, filename)
            try:
                temp_df = None
                for encoding in ['utf-8', 'CP949', 'EUC-KR']:
                    try:
                        temp_df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e_read_inner:
                        print(f"      경고: '{filename}' 파일을 읽는 중 예외 발생 (인코딩 외): {e_read_inner}. 건너뜁니다.")
                        temp_df = None
                        break
                        
                if temp_df is None:
                    print(f"      경고: '{filename}' 파일을 올바른 인코딩으로 읽을 수 없거나 내용이 없습니다. 건너뜝니다.")
                    continue
                # day = re.split(r"-|_", filename)[2]
                print(filename)
                if 'comments' in temp_df.columns:

                    # NaN 값은 빈 문자열로 처리하여 감성 분석에 포함시키지 않음
                    news_org_monthly_comments_list.extend(temp_df['comments'].fillna('').tolist())
                    news_org_monthly_comments_list = [comment for comment in news_org_monthly_comments_list if comment.strip()]
                    sentiment_df_org=analyze_sentiment_with_pipeline(news_org_monthly_comments_list)
                    mean_sentiment_org = sentiment_df_org.mean()
                    mean_sentiment_org_df=pd.DataFrame([mean_sentiment_org])
                    day = re.split(r"-|_", filename)[2]
                    mean_sentiment_org_df.insert(0,"year-month-day",month+"-"+day)
                    mean_sentiment_org_df.insert(0,"press_name",news_org)
                    all_data_for_df1=pd.concat([all_data_for_df1, mean_sentiment_org_df])



                else:
                    print(f"      경고: '{filename}' 파일에 'comments' 컬럼이 없습니다. 건너뜁니다.")
            except Exception as e:
                print(f"      오류: '{filename}' 파일을 처리하는 중 예외 발생: {e}. 건너뜁니다.")
    #일 저장
    all_data_for_df1.to_csv(f"../src/sentiment_analysis/{news_org}.csv", index=False)




    #     # 빈 댓글은 분석에서 제외 (모델 입력 시 오류 방지)
    #     news_org_monthly_comments_list = [comment for comment in news_org_monthly_comments_list if comment.strip()]
    #
    #     if not news_org_monthly_comments_list:
    #         print(f"      {month}월 {news_org}에 처리할 유효한 댓글이 없습니다. 건너뜝니다.")
    #         continue
    #
    #     print(f"      {month}월 {news_org}에 총 {len(news_org_monthly_comments_list)}개의 댓글을 통합했습니다. 감성 분석 시작...")
    #
    #     # analyze_sentiment_with_pipeline 함수는 이제 43가지 감성 점수를 담은 데이터프레임을 반환
    #     sentiment_df_org = analyze_sentiment_with_pipeline(news_org_monthly_comments_list)
    #
    #
    #     if sentiment_df_org.empty: # 감성 분석 결과 데이터프레임이 비어있는지 확인
    #         print(f"      {month}월 {news_org} 댓글에 대한 감성 분석 결과가 없습니다. 건너뜁니다.")
    #         continue
    #
    #     # 각 감성별 평균 점수 계산 (43가지 감성 모두에 대해)
    #     mean_sentiment_org = sentiment_df_org.mean()
    #
    #     # 전체 결과를 저장 (이후 필요할 때 활용 가능)
    #     all_results_by_month_and_org[month][news_org] = mean_sentiment_org
    #     # 워드클라우드용 댓글 데이터 추가 (옵션)
    #     # all_comments_for_wordcloud.extend(news_org_monthly_comments_list)
    #
    #     print(f"      {month}월 {news_org} 감성 분석 평균 점수 (확률):")
    #
    #     # 평균 점수가 높은 상위 5개 감성만 선택하여 출력
    #     top_5_emotions = mean_sentiment_org.sort_values(ascending=False).head(5)
    #
    #     day=re.split(r"-|_", filename)[2]
    #
    #     # print(filename)
    #     all_data_for_df = []
    #     for month, news_org_data in all_results_by_month_and_org.items():
    #         for news_org, sentiment_series in news_org_data.items():
    #             # Series를 Dictionary로 변환하고 'Month'와 'News_Org' 정보 추가
    #             row_data = sentiment_series.to_dict()
    #             row_data['Month'] = month
    #             row_data['News_Org'] = news_org
    #             row_data['day']=day
    #             all_data_for_df.append(row_data)
    #     if all_data_for_df:
    #         # 리스트를 DataFrame으로 변환
    #         final_df = pd.DataFrame(all_data_for_df)
    #
    #         # 컬럼 순서 재배열: 'Month', 'News_Org'를 맨 앞으로
    #         cols = ['Month','day' ,'News_Org'] + [col for col in final_df.columns if col not in ['Month','day', 'News_Org']]
    #         final_df = final_df[cols]
    #
    #         # 저장 경로 설정
    #         # base_dir은 이전에 정의한 데이터셋 경로 (예: 'C:\Users\bigba\Lecture\2025_1\Data_Analysis\Padebun\dataset')
    #         # output_filename = os.path.join(base_dir, 'overall_sentiment_analysis_results.csv')
    #         output_filename=f"./result/{news_org}-{month}_overall_sentiment_analysis_results.csv"
    #
    #         try:
    #             final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')  # 한글 깨짐 방지용 utf-8-sig
    #             print(f"통합 감성 분석 결과가 '{output_filename}'에 성공적으로 저장되었습니다.")
    #             print("\n저장된 파일 미리보기 (상위 5개 행):")
    #             print(final_df.head())
    #             break
    #         except Exception as e:
    #             print(f"통합 CSV 파일 저장 중 오류 발생: {e}")
    #     else:
    #         print("저장할 데이터가 없어 통합 CSV 파일을 생성하지 않습니다.")
    #
    #     print("-" * 50)
    #     if not top_5_emotions.empty:
    #         print(top_5_emotions)
    #         # break
    #
    #
    #     else:
    #         print("        분석된 감성 결과가 없습니다.")
    #         # break
    #     print("-" * 40)
    #
    #
    #
    # if not all_results_by_month_and_org[month]:
    #     print(f"경고: {month}월에 분석된 신문사 데이터가 없습니다.")
    #

print("\n--- 모든 신문사별 월별 데이터 처리 및 감성 분석 완료 ---")
