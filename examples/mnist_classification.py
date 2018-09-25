# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 16:53:58 2018

@author: l00467141
"""

# In[]
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
from sklearn.metrics import accuracy_score
import numpy as np 
from sklearn.model_selection import KFold,StratifiedKFold
# In[]
# 加载数据
digits=load_digits()
print(digits.data.shape)
data=digits.data
target = digits.target

X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.2)


print('Start training...')
# In[]
"""SVM"""
from sklearn import svm
clf_svm = svm.SVC(decision_function_shape='ovr',kernel='linear' )
# In[]
# 训练SVM模型
clf_svm.fit(X_train,y_train)
# In[]
# 对测试集预测
y_pred_smv = clf_svm.predict(X_test)
print(accuracy_score(y_test, y_pred_smv))
# In[]
# 训练集预测
y_train_svm = clf_svm.predict(X_train)
print(accuracy_score(y_train, y_train_svm))
# In[]
"""LR"""

from sklearn.linear_model import LogisticRegression
clf_LR=LogisticRegression()
# In[]
# 训练LR模型
clf_LR.fit(X_train,y_train)
# In[]
# 对测试集预测
y_pred_LR = clf_LR.predict(X_test)
print(accuracy_score(y_test, y_pred_LR))
# In[]
"""softmax"""
from sklearn.linear_model import LogisticRegression
clf_softmax=LogisticRegression(solver='sag',multi_class='multinomial')
# In[]
# 训练softmax模型
clf_softmax.fit(X_train,y_train)
# In[]
# 对测试集预测
y_pred_softmax = clf_softmax.predict(X_test)
# In[]
print(accuracy_score(y_test, y_pred_softmax))
# In[]
"""lightgbm"""
import lightgbm as lgb
# 创建模型，训练模型
#clf_lgb = lgb.LGBMClassifier(application='multiclass',boosting='gbdt',max_depth=6,num_leaves ='70')
clf_lgb = lgb.LGBMClassifier(application='multiclass',boosting='gbdt',max_depth=6,num_leaves ='10',min_data_in_leaf=10)

# In[]
clf_lgb.fit(X_train, y_train,eval_set=[(X_test, y_test)])
from sklearn.externals import joblib
joblib.dump(clf_lgb,'gbm.pkl')
# In[]
print('Start predicting...')
# 测试机预测
y_pred_lgb = clf_lgb.predict(X_test)
print(accuracy_score(y_test, y_pred_lgb))
# In[]
# 训练集预测
y_train_lgb = clf_lgb.predict(X_train)
print(accuracy_score(y_train, y_train_lgb))
