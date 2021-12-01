'''
예시문제 2번 
이번에는 레이블 인코딩을 적용하지 않고 범주형 변수를 제거하여 모델을 학습시킨다.
세 가지 모델 Logistic Regression, Neural Network, Random Forest의 학습을 통해
예측한 확률값을 비교해보고 소프트 보팅 앙상블하여 제출한다.
'''

# 데이터 불러오기
import pandas as pd
X_train = pd.read_csv("data/X_train.csv", encoding='cp949') # 한글 포함된 경우
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv", encoding='cp949')


# 범주형 변수 탐색 및 전처리
print(X_train['주구매지점'].unique())
print(len(X_train['주구매지점'].unique()))
# print(X_train['주구매지점'].value_counts())
print(X_train['주구매상품'].unique())
print(len(X_train['주구매상품'].unique()))
# print(X_train['주구매상품'].value_counts())
'''
'주구매지점'과 '주구매상품'은 범주 종류가 각각 24개, 42개로 많고 순서가 존재하지 않는다.
1. 원-핫 인코딩 적용시 다중공선성 문제와 메모리 비효율성 이슈가 발생한다.
2. 레이블 인코딩 적용시 숫자의 크고 작은 특성이 반영되므로 학습에 방해된다.
따라서, 두 변수 모두 인코딩이 적절하지 않으므로 제거한다.
('주구매상품'은 성별과 관련성이 크므로 만약 모델 성능이 낮을 시 피처 엔지니어링을 추가하자.)

또한, 고객ID도 성별과 관련이 없어서 학습에 방해되므로 제거한다.
''' 
drop_cols = ['주구매지점', '주구매상품', 'cust_id']
X_train.drop(drop_cols, axis=1, inplace=True)

cust_id = X_test.loc[:,'cust_id']  # index column
X_test.drop(drop_cols, axis=1, inplace=True)

# y_train.drop('cust_id', axis=1, inplace=True)으로 했더니 fit_transform에서 에러 발생
# fit_transform(X,y)에서 y는 1d-array여야 하는데 현재 y_train의 shape은 (3500,1)인 데이터프레임이다.
# 따라서, 성별 컬럼만 y_train으로 가져온다.
y_train = y_train['gender'] # Series, shape=(3500,)
assert y_train.ndim == 1


# 데이터 정제: 결측값 처리
X_train.loc[:,'환불금액'].fillna(0, inplace=True)
X_test.loc[:,'환불금액'].fillna(0, inplace=True)


# 모델 학습 및 예측
# =========  1. Logistic Regression  ===========
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f'Logistic Regression Accuracy: {lr.score(X_train, y_train)}')
pred_lr = lr.predict_proba(X_test) # (2482, 2) 
pred_lr = pred_lr[:,1]  # 남자일 확률


#============  2. Neural Network  ==============
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
print(f'MLP Accuracy: {mlp.score(X_train, y_train)}')
pred_mlp = mlp.predict_proba(X_test)
pred_mlp = pred_mlp[:,1]


#==========  3. RandomForest  ===========
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=20, max_depth=3, max_samples=0.1, random_state=999)
rf.fit(X_train, y_train)
print(f'RandomForest Accuracy: {rf.score(X_train, y_train)}')
pred_rf = rf.predict_proba(X_test)
pred_rf = pred_rf[:,1]


# 모델 성능 비교
output = pd.DataFrame({'cust_id': cust_id, 'LR': pred_lr, 'MLP': pred_mlp, 'RF': pred_rf})
print(output.head(10))
'''
Logistic Regression Accuracy: 0.624
MLP Accuracy: 0.5651428571428572
RandomForest Accuracy: 0.6408571428571429

   cust_id        LR  MLP        RF
0     3500  0.425232  0.0  0.426166
1     3501  0.198986  0.0  0.189003
2     3502  0.238942  1.0  0.217317
3     3503  0.490611  0.0  0.439881
4     3504  0.497643  0.0  0.463639
5     3505  0.489186  0.0  0.431278
6     3506  0.150611  0.0  0.202558
7     3507  0.498297  0.0  0.470604
8     3508  0.492867  0.0  0.458789
9     3509  0.362807  0.0  0.236630

- MLP 모델의 예측값은 거의 0이다..!! 뭔가 이상하다. 확률이 아니라 분류 클래스 값을 리턴한 걸까?? MLP 모델링 다시 보자.
- LR와 RF의 결과값은 대체로 비슷하다. 다만, RF는 오버피팅의 위험이 있으므로 앙상블하여 제출한다.
- 세 모델의 학습 정확도는 범주형 변수를 레이블 인코딩하여 반영했을 때와 동일하다. 두 개의 범주형 변수가 학습에 도움이 안 되는 것으로 판단.
'''

# 답안 제출
pred = (pred_lr + pred_rf)/2  # 앙상블 - 소프트 보팅
pd.DataFrame({'cust_id': cust_id, 'gender': pred}).to_csv('1234.csv', index=False)
