'''
제공 데이터
y_train.csv: 고객의 성별 데이터 (학습용)
    - cust_id (고객ID)
    - gender (0:여자, 1:남자)
X_train.csv, X_test.csv: 고객의 상품구매 속성 (학습용 및 평가용)
    - cust_id: int64
    - 총구매액: int64
    - 최대구매액: int64
    - 환불금액: float64
    - 주구매상품: object
    - 주구매지점: object
    - 내점일수: int64
    - 내점당구매건수: float64
    - 주말방문비율: float64
    - 구매주기: int64

X_train, y_train: 3500명 / X_test: 2482명
'''

# 데이터 불러오기
import pandas as pd
X_train = pd.read_csv("data/X_train.csv", index_col=0)
y_train = pd.read_csv("data/y_train.csv")
y_train = y_train['gender']
print(y_train.value_counts())

X_test = pd.read_csv("data/X_test.csv")
X_test_id = X_test.loc[:,'cust_id']
X_test = X_test.iloc[:,1:]


# 데이터 전처리: 라벨 인코딩, 결측값 처리
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train.loc[:,['주구매상품','주구매지점']] = X_train.loc[:,['주구매상품','주구매지점']].apply(le.fit_transform)
X_test.iloc[:,[3,4]] = X_test.iloc[:,[3,4]].apply(le.fit_transform)

X_train.loc[:,'환불금액'] = X_train.loc[:,'환불금액'].fillna(0)
X_test.iloc[:,2] = X_test.iloc[:,2].fillna(0)


# 모델 학습
#============  1. Logistic Regression =============
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(f'Logistic Regression Accuracy: {model.score(X_train, y_train)}')

#============  2. Neural Network  ==============
from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
model.fit(X_train, y_train)
print(f'MLP Accuracy: {model.score(X_train, y_train)}\n')

#============  3. RandomForest  ================
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimator=20, max_depth=3, max_samples=0.1, random_state=999)
model.fit(X_train, y_train)
print(f'RandomForest Accuracy: {model.score(X_train, y_train)}')


# 테스트 데이터 예측
pred = model.predict_proba(X_test) # (2482, 2) 
pred = pred[:,1]  # 남자일 확률


# 답안 제출
pd.DataFrame({'cust_id': X_test_id, 'gender': pred}).to_csv('1234.csv', index=False)
