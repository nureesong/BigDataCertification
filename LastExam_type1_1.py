'''
[빅데이터분석기사 실기 2회 기출문제] 
작업형 제1유형 문제 1번
- f5 컬럼을 기준으로 상위 10개의 데이터를 구하고
- f5 컬럼 10개 중 최솟값으로 데이터를 대체한 후
- age 컬럼에서 80 이상인 데이터의 f5 컬럼 평균값을 계산하시오.
'''

# 데이터 로드 및 탐색
import pandas as pd
df = pd.read_csv('./data/basic1.csv')
# df.head()
# df.info()
# df.describe()

# 1. 상위 10개 데이터 구하기
f5_top10 = df.nlargest(10, 'f5')
# df.sort_values('f5', ascending=False).head(10)  # 다른 방법

# 2. 10개 중 최솟값으로 데이터 대체하기
df.loc[f5_top10.index, 'f5'] = f5_top10['f5'].min()
# print(df.nlargest(15, 'f5')) # 잘 대체되었는지 확인

# 3. age가 80이상인 데이터
age80 = df.loc[df['age']>=80, :]
answer = age80['f5'].mean()
print(f'정답: {round(answer, 4)}')
