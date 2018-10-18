# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:55:54 2018

@author: l00467141
"""

# In[]

from sklearn.datasets import load_svmlight_file
import lightgbm as lgb

# In[]
X_train, y_train = load_svmlight_file("./rank.train")
X_test,y_test=load_svmlight_file("./rank.test")
# In[]
f = open("./rank.train.query","r",encoding='UTF-8')   #设置文件对象
qid_train_count_list = f.readlines()  #直接将文件中按行读到list里
f.close()             #关闭文
# In[]
#for i in range(len(qid_train_count_list)):
#    qid_train_count_list[i]=int(qid_train_count_list[i])
# In[]
f = open("./rank.test.query","r",encoding='UTF-8')   #设置文件对象
qid_test_count_list = f.readlines()  #直接将文件中按行读到list里
f.close()             #关闭文
# In[]
#for i in range(len(qid_test_count_list)):
#    qid_test_count_list[i]=int(qid_test_count_list[i])
# In[]
lgb_train = lgb.Dataset(X_train, y_train,group=qid_train_count_list)
lgb_eval = lgb.Dataset(X_test, y_test,group=qid_test_count_list)
params={
        "config":"./train.conf"
        }

print('Start training...')
# train
gbm = lgb.train(params=params,train_set=lgb_train)
#print('Start training...')
#gbm=lgb.Booster(model_file='lambdarank_example.model')
# In[]
print('Start predicting...')
# predict
prob_pred_lgb = gbm.predict(X_test,config="predict.conf")
# In[]
gbm.save_model("lambdarank_example_model.txt")