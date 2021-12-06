'''
[빅데이터분석기사 실기 2회 기출문제] 
작업형 제1유형 문제 3번

이상값 검출
- 데이터셋의 age 컬럼의 이상치를 더하시오.
- 단, 평균으로부터 '표준편차 * 1.5'를 벗어나는 영역을 이상치라고 판단함.
'''

# 데이터 로드 및 age 컬럼 탐색
import pandas as pd
df = pd.read_csv('./data/basic1.csv')
df['age'].describe()

# 하한, 상한 구하기
m, s = df['age'].mean(), df['age'].std()
print(f'age 컬럼의 평균: {round(m,2)}, 표준편차: {round(s,2)}')
low, high = m - 1.5*s, m + 1.5*s
print(f'정상 데이터 범위: ({round(low,2)}, {round(high,2)})')

# 이상치 검출
df_low = df.loc[df['age'] < low, 'age']
df_high = df.loc[df['age'] > high, 'age']
print(f'이상치로 판단된 데이터는 총 {len(df_low) + len(df_low)}개이다.\n')

# 정답 출력
print(f'정답: {pd.concat([df_low, df_high]).sum()}')