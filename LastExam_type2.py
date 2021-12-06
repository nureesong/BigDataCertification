''' 
[빅데이터분석기사 실기 2회 기출문제] 작업형 제2유형
다음은 전자상거래 배송 데이터이다. 👉🏻 10999 observations of 12 variables.
고객 10999명에 대한 데이터를 이용하여 물건의 정시 도착 여부(1: No, 0: Yes)에 대한 예측 모형을 만든 후,
이를 평가용 데이터에 적용하여 얻은 제품이 ✨정시에 도착할 확률✨을 다음과 같은 형식의 csv 파일로 생성하시오.

ID, Reached.on.Time_Y.N
3500, 0.267
3501, 0.578
3502, 0.885

🚨 유의사항
- 성능이 우수한 예측모형을 구축하기 위해서는 적절한 데이터 전처리, 피처 엔지니어링, 분류 알고리즘, 하이퍼파라미터 튜닝, 모형 앙상블 등이 수반되어야 한다.
- 제출한 모델의 성능은 ROC-AUC 평가지표에 따라 채점한다.
- 수험번호.csv파일이 만들어지도록 코드를 제출한다.
  pd.DataFrame({'ID': X_test.ID, 'Reached.on.Time_Y.N': pred}).to_csv('0030.csv', index=False)

📌 To Do List
- 전처리: Label encoding, 이상치/결측치 처리, Normalization
- 모델링: XGBoost, SVM, RandomForest, Logistic Regression, MLP
'''

''' Train, Test 데이터 분리하기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('/data/Ecommerce_shipping_Train.csv')
X_train, X_test = train_test_split(df, test_size=0.2, random_state=2021)
y_train = X_train[['ID', 'Reached.on.Time_Y.N']]
X_train.drop(columns=['Reached.on.Time_Y.N'], inplace=True)

y_test = X_test[['ID', 'Reached.on.Time_Y.N']]
X_test.drop(columns=['Reached.on.Time_Y.N'],inplace=True)

# Index Re-numbering, ID 컬럼은 그대로.
X_train.set_index(keys=np.arange(len(X_train)), inplace=True)
y_train.set_index(keys=np.arange(len(y_train)), inplace=True)
X_test.set_index(keys=np.arange(len(X_test)), inplace=True)

X_train.to_csv('data/Ecommerce_X_train.csv', index=False)
y_train.to_csv('data/Ecommerce_y_train.csv', index=False)
X_test.to_csv('data/Ecommerce_X_test.csv', index=False)
y_test.to_csv('data/Ecommerce_y_test.csv', index=False)
'''

import pandas as pd
X_train = pd.read_csv('data/Ecommerce_X_train.csv')
y_train = pd.read_csv('data/Ecommerce_y_train.csv')
X_test = pd.read_csv('data/Ecommerce_X_test.csv')

y_train = y_train.iloc[:,-1]
assert y_train.ndim == 1  # y_train은 1차원 배열이어야 한다.
print('타겟변수(정시도착여부) 탐색')
print(y_train.value_counts())

# ID 컬럼 제거
X_train.drop(columns=['ID'], inplace=True)
X_test_id = X_test['ID']
X_test.drop(columns=['ID'], inplace=True)

# 범주형 변수 탐색
categorical = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
for col in categorical:
    print(f'\n{col} 컬럼 탐색')
    print(X_train[col].value_counts())


# ======  데이터 전처리 (Lable Encoding)  =======
# Warehouse_block - {A:0, 1:B, 2:C, 3:D, 4:F} / Gender - {F:0, M:1}
# Mode_of_Shipment - {Flight:0, Ship:1, Road:2} / Product_importance - {low:0, medium:1, high:2}

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X_train[['Warehouse_block','Gender']] = X_train[['Warehouse_block','Gender']].apply(le.fit_transform)
X_test[['Warehouse_block','Gender']] = X_test[['Warehouse_block','Gender']].apply(le.fit_transform)

shipment = {'Flight':0, 'Ship':1, 'Road':2}
X_train['Mode_of_Shipment'] = [shipment[i] for i in X_train['Mode_of_Shipment']]
X_test['Mode_of_Shipment'] = [shipment[i] for i in X_test['Mode_of_Shipment']]

importance = {'low':0, 'medium':1, 'high':2}
X_train['Product_importance'] = [importance[i] for i in X_train['Product_importance']]
X_test['Product_importance'] = [importance[i] for i in X_test['Product_importance']]


# =======  모델 학습 및 평가  =========
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

print('\n==== Accuracy ====')
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f'LR: {round(lr.score(X_train, y_train), 2)}')
pred_lr = lr.predict_proba(X_test)
pred_lr = pred_lr[:,0] # 정시에 도착(0으로 예측)할 확률


mlp = MLPClassifier(hidden_layer_sizes=(100,),
                    learning_rate_init=1e-4,
                    random_state=42)
mlp.fit(X_train, y_train)
print(f'MLP: {round(mlp.score(X_train, y_train), 2)}')
pred_mlp = mlp.predict_proba(X_test)
pred_mlp = pred_mlp[:,0]


rf = RandomForestClassifier(n_estimators=15,
                            max_depth=3,
                            max_samples=0.2,
                            random_state=42)
rf.fit(X_train, y_train)
print(f'RF: {round(rf.score(X_train, y_train), 2)}')
pred_rf = rf.predict_proba(X_test)
pred_rf = pred_rf[:,0]


knn = KNeighborsClassifier(n_neighbors=5,
                           leaf_size=10)
knn.fit(X_train, y_train)
print(f'KNN: {round(knn.score(X_train, y_train), 2)}')  # overfitting!!
pred_knn = knn.predict_proba(X_test)
pred_knn = pred_knn[:,0]


svc = SVC(probability=True, # 반드시 해야함!
          kernel='rbf',
          random_state=42)
svc.fit(X_train, y_train)
print(f'SVM: {round(svc.score(X_train, y_train), 2)}')  # 데이터 10000개 이상이면 느려서 비효율적
pred_svc = svc.predict_proba(X_test)
pred_svc = pred_svc[:,0]


xgb = XGBClassifier(n_estimators=100,
                    max_depth=3,
                    use_label_encoder=False,
                    eval_metric='logloss')
xgb.fit(X_train, y_train)
print(f'XGB: {round(xgb.score(X_train, y_train), 2)}')
pred_xgb = xgb.predict_proba(X_test)
pred_xgb = pred_xgb[:,0]

print('\n========  예측결과(정시에 도착할 확률) 비교  ========')
output = pd.DataFrame({'ID':X_test_id, 'LR':pred_lr, 'MLP':pred_mlp, 'RF':pred_rf, 'KNN':pred_knn, 'SVM':pred_svc, 'XGB':pred_xgb})
print(output.head(5))

# 정답 제출
pred = xgb.predict(X_test)  # xgb의 성능이 제일 좋다.
y_pred = pd.DataFrame({'ID':X_test_id, 'Reached.on.Time_Y.N':pred})
# y_pred.to_csv('data/003002225.csv', index=False)


# =======  실제 정답률  =======
import numpy as np
y_test = pd.read_csv('data/Ecommerce_y_test.csv')
acc = np.mean(y_pred['Reached.on.Time_Y.N'].values == y_test['Reached.on.Time_Y.N'].values)
print(f'\n XGB 모델로 예측한 결과 >> 실제 정답률: {round(acc,2)}')