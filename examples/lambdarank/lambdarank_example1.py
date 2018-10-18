# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:18:07 2018

@author: l00467141
"""

# In[]

from sklearn.datasets import load_svmlight_file
import lightgbm as lgb
import os,sys
from sklearn.datasets import load_svmlight_file
# In[]
#查看当前工作目录
print("当前的工作目录为：%s" %os.getcwd())
# In[]
"""获取训练集的qid"""
f = open("./Fold1/trainingset.txt","r",encoding='UTF-8')   #设置文件对象
train_list = f.readlines()  #直接将文件中按行读到list里
f.close()             #关闭文件
# In[]
X_train,y_train=load_svmlight_file("data_train.txt")
X_test,y_test=load_svmlight_file("data_test.txt")
# In[]
f = open("data_train_group.txt","r",encoding='UTF-8')
data_group_train = f.readlines()
f.close()
# In[]
f = open("data_test_group.txt","r",encoding='UTF-8')
data_group_test = f.readlines()
f.close()
# In[]
lgb_train = lgb.Dataset(X_train, y_train,group=data_group_train)
lgb_eval = lgb.Dataset(X_test, y_test,group=data_group_test)
params={
        "application":"lambdarank",
        "boosting_type": "gbdt",
        "metric":"ndcg",
        "max_bin" : "255",
        "num_trees":"100",
        }

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train)
gbm.booster_.save_model("lambdarank_example_model1.txt")
# In[]
print('Start predicting...')
# predict
prob_pred_lgb = gbm.predict(X_test)
# In[]








