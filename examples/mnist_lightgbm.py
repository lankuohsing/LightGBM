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
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
# In[]
"""加载数据集"""
digits=load_digits()
print(digits.data.shape)
data=digits.data
target = digits.target
X=data
y=target
# In[]
n_splits=5
y_pred_lgb_list=[]
y_test_list=[]
sfolder = StratifiedKFold(n_splits=n_splits,random_state=0,shuffle=True)
for train_index, test_index in sfolder.split(X,y):
    # In[]
    X_train=X[train_index]
    y_train=y[train_index]
    #y_train=y_train.reshape(y_train.shape[0],1)
    X_test=X[test_index]
    y_test=y[test_index]
   # y_test=y_test.reshape(y_test.shape[0],1)
    """lightgbm"""
    # create dataset for lightgbm
    # In[]
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'max_depth':'6',
        'num_leaves' :'10',
        'min_data_in_leaf':'10',
        'num_class':10
    }
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train)
    # In[]
    print('Start predicting...')
    # predict
    prob_pred_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # In[]
    y_pred_lgb=np.where(prob_pred_lgb==np.max(prob_pred_lgb,axis=1).reshape(prob_pred_lgb.shape[0],1))[1].reshape(prob_pred_lgb.shape[0],1)
    y_pred_lgb_list.append(y_pred_lgb.reshape(y_pred_lgb.shape[0],1))
    y_test_list.append(y_test.reshape(y_test.shape[0],1))

# In[]
y_pred_lgb_np=y_pred_lgb_list[0]
y_test_np=y_test_list[0]
for i in range(1,len(y_pred_lgb_list)):
    y_pred_lgb_np=np.vstack((y_pred_lgb_np,y_pred_lgb_list[i]))
    y_test_np=np.vstack((y_test_np,y_test_list[i]))
# In[]
print("result:",classification_report(y_true=y_test_np, y_pred=y_pred_lgb_np))
# In[]
'''
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.2)
# In[]
"""开始"""
print('Start training...')
# In[]
"""lightgbm"""

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# In[]
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth':'6',
    'num_leaves' :'10',
    'min_data_in_leaf':'10',
    'num_class':10
}
# In[]
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model_mnist_lgb.txt')
# In[]
print('Start predicting...')
# predict
prob_pred_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# In[]
y_pred_lgb=np.where(prob_pred_lgb==np.max(prob_pred_lgb,axis=1).reshape(prob_pred_lgb.shape[0],1))[1].reshape(prob_pred_lgb.shape[0],1)
# In[]
# eval
print('The accuracy of prediction is:', accuracy_score(y_test, y_pred_lgb))
# In[]
'''