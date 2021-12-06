'''
[빅데이터분석기사 실기 2회 기출문제] 
작업형 제1유형 문제 2번

- 데이터셋의 앞에서부터 순서대로 80% 데이터만 활용해서
- f1 컬럼 결측치를 중앙값으로 채우기 전후의 표준편차를 구하고
- 두 표준편차 차이를 계산하시오.
'''

# 데이터 로드
import pandas as pd
df = pd.read_csv('./data/basic1.csv')

# 1. 80% 데이터만 추출하기
df = df.iloc[:int(len(df)*0.8), :]
# print(df.tail()) # 80개만 잘 가져왔는지 확인한다.

# 2. f1 컬럼 탐색
f1 = df['f1']
print(f'f1 컬럼에는 {f1.isnull().sum()}개의 결측값이 존재한다.')
med = f1.median()
print(f'f1 컬럼의 중앙값: {med}\n')

# 3. 표준편차 비교
std = f1.std()
print(f'결측값을 채우기 전 f1 컬럼의 표준편차: {round(std, 4)}')
std_fill = f1.fillna(med).std()
print(f'결측값을 채운 후 f1컬럼의 표준편차: {round(std_fill, 4)}\n')

# 정답 출력
print(f'정답: {round(std - std_fill, 4)}')