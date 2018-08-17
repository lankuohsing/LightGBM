# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:29:18 2018

@author: l00467141
"""
# In[]
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
from sklearn.metrics import accuracy_score
# In[]
# 加载数据
print('Load data...')

iris = load_iris()
data=iris.data
target = iris.target
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.2)

# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

print('Start training...')

# In[]
# 创建模型，训练模型
gbm = lgb.LGBMClassifier(objective='multiclass',num_class=3)
# In[]
gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)])
# In[]
print('Start predicting...')
# 测试机预测
y_pred = gbm.predict(X_test)
# In[]

accuracy_score(y_test, y_pred)
# In[]
