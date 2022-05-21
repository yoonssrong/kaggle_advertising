import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, plot_confusion_matrix



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male','month','day','hour','Country','region','region_incomeLevel']]  # 독립변수
Y_train = train_data['Clicked on Ad']      # 알고자하는 종속변수

X_test = test_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male','month','day','hour','Country','region','region_incomeLevel']]  # 독립변수
Y_test = test_data['Clicked on Ad']      # 알고자하는 종속변수

from sklearn.preprocessing import StandardScaler

# Standardization 평균 0 / 분산 1
scaler = StandardScaler()

# 교차검증시
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


dtrain = xgb.DMatrix(data=X_train, label = Y_train)
dtest = xgb.DMatrix(data=X_test, label=Y_test)

# 모델 생성
# num_boost_round 만큼 반복하는데 early_stopping_rounds 만큼 성능 향상이 없으면 중단
# early_stopping_rounds를 사용하려면 eval 데이터 셋을 명기해야함
params = {'max_depth' : 3,
         'eta' : 0.1,
         'objective' : 'binary:logistic',
         'eval_metric' : 'logloss',
         'early_stoppings' : 100 }

xgb_model = xgb.train(params = params, dtrain = dtrain, num_boost_round = 400,
                        early_stopping_rounds = 100, evals=[(dtrain,'train'),(dtest,'eval')])

pred_probs = xgb_model.predict(dtest)
print('predict() 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨')
print(np.round(pred_probs[:10], 3))

# 예측 확률이 0.5보다 크면 1, 그렇지 않으면 0으로 예측값 결정해 리스트 객체인 preds에 저장
preds = [ 1 if x > 0.5 else 0 for x in pred_probs]
print('예측값 10개만 표시: ', preds[:10])

# 혼동행렬, 정확도, 정밀도, 재현율, F1, AUC 불러오기
def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    print('AUC: {:.4f}'.format(AUC))

get_clf_eval(Y_test, preds)

confusion = confusion_matrix(Y_test, preds)
print(confusion)

import matplotlib.pyplot as plt

title = None
cmap=plt.cm.Blues
plt.figure(figsize=(4, 4))
plt.imshow(confusion, interpolation='nearest', cmap=cmap)  # , cmap=plt.cm.Greens
plt.title(title, size=12)
plt.colorbar(fraction=0.05, pad=0.05)
tick_marks = np.arange(2, 2)
plt.xticks(np.arange(2), ('0', '1'))
plt.yticks(np.arange(2), ('0', '1'))
plt.show()