import time
start = time.time()
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male','month','day','hour','Country','region','region_incomeLevel']]  # 독립변수
Y_train = train_data['Clicked on Ad']      # 알고자하는 종속변수

X_test = test_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male','month','day','hour','Country','region','region_incomeLevel']]  # 독립변수
Y_test = test_data['Clicked on Ad']      # 알고자하는 종속변수


lgb = lgb.LGBMClassifier()#Define the parameters
parameters = {'num_leaves':[20,40,60,80,100],
              'min_child_samples':[5,10,15],
              'max_depth':[-1,5,10,20],
              'learning_rate':[0.05,0.1,0.2],
              'reg_alpha':[0,0.01,0.03]}  #Define the scoring

clf = GridSearchCV(lgb,parameters,scoring='accuracy')
clf.fit(X=X_train, y=Y_train)
print(clf.best_params_)
predicted = clf.predict(X_test)
print('Classification of the result is:')

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

get_clf_eval(Y_test, predicted)

end = time.time()
print('Execution time is:')
print(end - start)
