'''
[빅데이터분석기사 실기 3회 연습문제] 
작업형 제1유형

- mtcars 데이터셋의 qsec 컬럼을 최소최대 척도(Min-Max Scale)로 변환한 후
- 0.5보다 큰 값을 가지는 레코드 수를 구하시오.
'''

import pandas as pd
df = pd.read_csv("data/mtcars.csv")


print("\n======= From scratch =======")
qsec = df.loc[:, 'qsec']  # pandas Series
qsec_min, qsec_max = qsec.min(), qsec.max()
print(f'qsec 컬럼의 최솟값: {qsec_min}, 최댓값: {qsec_max}')

qsec_scaled = [ (i - qsec_min)/(qsec_max - qsec_min) for i in qsec if (i - qsec_min)/(qsec_max - qsec_min) > 0.5 ]
print(f'0.5보다 큰 값을 가지는 레코드 수: {len(qsec_scaled)}')
# qsec_scaled > 0.5 -> Error!! list이므로 비교 연산 불가


print("\n======= Scikit-learn =======")
print(f'shape of qsec: {qsec.shape}')  # (32,) 1d-array

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
fitted = scaler.fit_transform(qsec.values.reshape(-1,1))  # (32,1) 2d-array
print(f'0.5보다 큰 값을 가지는 레코드 수: {sum(fitted.reshape(-1) > 0.5)}')
# fitted.reshape(-1)은 1d-array이므로 비교 연산 가능