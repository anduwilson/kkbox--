import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
df_test=pd.read_csv('test.csv')
cat_drop=['source_system_tab','source_screen_name','source_type']
df_test.drop(cat_drop,axis=1)
n_test_num=df_test.shape[0]
#分数列表
score_list=list()
#id
ids=df_test['id']

df=pd.DataFrame()
df['id']=ids

import json
#读取模型
def load_json(filepath):
    with open(filepath) as file:
        dict_=json.load(file)
        mu=dict_['mu']
        K=dict_['K']
        bi=dict_['bi']
        bu=dict_['bu']
        pu=dict_['pu']
        qi=dict_['qi']

    return K,mu,bi,bu,pu,qi
K,mu,bi,bu,pu,qi=load_json('save_model.json')

#加载训练好的svd模型
def svd_pred(uid,iid):
    score=mu+bi[iid]+bu[uid]+np.sum(qi[iid]*pu[uid])

    return score

#取出每一行中的user_id song_id，输入进svd_pred()
for i in range(n_test_num):
    user_id=df_test.iloc[i]['msno']
    item_id=df_test.ilov[i]['song_id']

    test_score=svd_pred(user_id,item_id)
    score_list.append(test_score)

df['score']=score_list

df.to_csv('submission_svd_cf_pred.csv',float_format='%.5f')