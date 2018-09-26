# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:18:07 2018

@author: l00467141
"""

# In[]

from sklearn.datasets import load_svmlight_file
import lightgbm as lgb
# In[]
# In[]
"""获取训练集的qid"""
f = open("./Fold1/trainingset.txt","r",encoding='UTF-8')   #设置文件对象
train_list = f.readlines()  #直接将文件中按行读到list里
f.close()             #关闭文件
qid_train_list=[]
for i in range(len(train_list)):
    qid_train_i=int(train_list[i].split(" ")[1].split(":")[1])
    qid_train_list.append(qid_train_i)
# In[]
qid_train_set=set(qid_train_list)
# In[]
qid_train_count_list=[]
# In[]
for qid_train in qid_train_set:
    qid_train_count=qid_train_list.count(qid_train)
    qid_train_count_list.append(qid_train_count)
# In[]
with open("./Fold1/trainingset.txt.query","w") as f:
    for qid_count in qid_train_count_list:
        f.write(str(qid_count))
        f.write("\n")
    print("加载入文件完成...")
# In[]
# In[]
"""获取测试集的qid"""
f = open("./Fold1/testset.txt","r",encoding='UTF-8')   #设置文件对象
test_list = f.readlines()  #直接将文件中按行读到list里
f.close()             #关闭文件
qid_test_list=[]
for i in range(len(test_list)):
    qid_test_i=int(test_list[i].split(" ")[1].split(":")[1])
    qid_test_list.append(qid_test_i)
# In[]
qid_test_set=set(qid_test_list)
# In[]
qid_test_count_list=[]
# In[]
for qid_test in qid_test_set:
    qid_test_count=qid_test_list.count(qid_test)
    qid_test_count_list.append(qid_test_count)
# In[]
with open("./Fold1/testset.txt.query","w") as f:
    for qid_count in qid_test_count_list:
        f.write(str(qid_count))
        f.write("\n")
    print("加载入文件完成...")
# In[]
X_train, y_train = load_svmlight_file("./Fold1/trainingset.txt")
X_test,y_test=load_svmlight_file("./Fold1/testset.txt")
# In[]
lgb_train = lgb.Dataset(X_train, y_train,group=qid_train_count_list)
lgb_eval = lgb.Dataset(X_test, y_test,group=qid_test_count_list)
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
# In[]
print('Start predicting...')
# predict
prob_pred_lgb = gbm.predict(X_test)















# In[]
#clf_lgb = lgb.LGBMClassifier(application='multiclass',boosting='gbdt',max_depth=6,num_leaves ='10',min_data_in_leaf=10)

# In[]
#clf_lgb.fit(X_train, y_train,eval_set=[(X_test, y_test)])


# In[]
