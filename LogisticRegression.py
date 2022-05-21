import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


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


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', max_iter=1000)  # 모델이라는 이름아래 LogisticRegression 함수를 장착
model.fit(X_train, Y_train)

pred = model.predict(X_test)

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

get_clf_eval(Y_test, pred)

confusion = confusion_matrix(Y_test, pred)
print(confusion)

import matplotlib.pyplot as plt

title = None
cmap=plt.cm.Reds
plt.figure(figsize=(4, 4))
plt.imshow(confusion, interpolation='nearest', cmap=cmap)  # , cmap=plt.cm.Greens
plt.title(title, size=12)
plt.colorbar(fraction=0.05, pad=0.05)
tick_marks = np.arange(2, 2)
plt.xticks(np.arange(2), ('0', '1'))
plt.yticks(np.arange(2), ('0', '1'))
plt.show()