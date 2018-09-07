# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 21:08:54 2018

@author: l00467141
"""

# In[]
"""导入包"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# In[]
"""加载数据集"""
digits=load_digits()
print(digits.data.shape)
data=digits.data
target = digits.target
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.2)
# In[]
print('Start training...')
# In[]
"""lightgbm"""
import lightgbm as lgb
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# In[]
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': {'multi_error'},
    #'num_leaves': 31,
    #'learning_rate': 0.05,
    #'feature_fraction': 0.9,
    #'bagging_fraction': 0.8,
    #'bagging_freq': 5,
    #'verbose': 0
    'num_class': 10
}
# In[]
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model_mnist_lgb.txt')
# In[]
print('Start predicting...')
# predict
y_pred_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', accuracy_score(y_test, y_pred_lgb))
