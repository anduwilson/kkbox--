import numpy as np
import pandas as pd
import scipy.sparse as ss
import pickle
import scipy.io as sio


pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)
#pd.set_option('display.max_rows',30)
#
df_train=pd.read_csv('train.csv')
print('用户：',df_train['msno'].unique().shape[0])
print('歌曲：',df_train['song_id'].nunique())


#---------------------------数据预处理-----------------------------------------------
#type={'msno','song_id','source_system_tab','source_screen_name','source_type','target'}
#去掉'source_system_tab','source_screen_name','source_type'三种不相关的特征
df_train=df_train.drop(['source_system_tab','source_screen_name','source_type'],axis=1)
print('去掉无用的特征后训练集维度：',df_train.shape)


#----------------------------特征工程-------------------------------------------------
#msno song_id属于hash编码后的特征
#计算用户对每首歌曲的打分
df_user_rating=df_train[['msno','target']].groupby('msno').sum().reset_index()

df_user_rating.rename(columns={'target':'total_rating'},inplace=True)
#print(df_user_rating,)


#每首歌曲的播放比例
df_train=pd.merge(df_train,df_user_rating)
del df_user_rating
#print('用户订阅过的音乐，及总和：\n',df_train)


#删除总打分次数为0的用户（这里打分此时为0，代表着该用户在本月是第一次来
# 或者该用户上个月订阅过音乐，但是这个月该用户流失了
#通过观察，发现索引为7377417的用户订阅的音乐次数为0，所以去掉该用户
#total_rating为0的索引
index=df_train[df_train.total_rating==0].index.tolist()
#print('index=',len(index))
df_train=df_train.drop(index=index)


print(df_train.sort_values(by=['total_rating'],ascending=False))
df_train['fractional_rating_count']=df_train['target']/df_train['total_rating']

#print(df_train)

#所有的用户和item
users=df_train['msno'].unique()
n_users=len(users)
print('用户数量为：',n_users)
items=df_train['song_id'].unique()
n_items=len(items)
print('歌曲数量为：',n_items)

#计算倒排表
from collections import defaultdict
user_items=defaultdict(set)
item_users=defaultdict(set)

#用户-物品关系矩阵
user_item_scores=ss.dok_matrix((n_users,n_items))

#重新编码用户索引
users_index=dict()
#物品索引
items_index=dict()

for i,u in enumerate(users):
    users_index[u]=i
for i,j in enumerate(items):
    items_index[j]=i
n_records=df_train.shape[0]
print('n_records=',n_records)



for i in range(n_records):
    user_index_i=users_index[df_train.iloc[i]['msno']]#用户id对应的索引
    item_index_i=items_index[df_train.iloc[i]['song_id']]#歌曲id对应的索引

    #倒排表
    user_items[user_index_i].add(item_index_i)# user_index_I所对应的所有音乐
    item_users[item_index_i].add(user_index_i)#song_id的所有订阅该音乐的用户

    #分数
    score=df_train.iloc[i]['fractional_rating_count']
    user_item_scores[user_index_i,item_index_i]=score

#保存倒排表
pickle.dump(user_items,open('user_items.pkl','wb'))
pickle.dump(item_users,open('item_users.pkl','wb'))

#保存用户-物品关系矩阵,以备后用
sio.mmwrite('user_item_scores',user_item_scores)

#保存用户、物品索引表
pickle.dump(users_index,open('users_index.pkl','wb'))
pickle.dump(items_index,open('items_index.pkl','wb'))


#-------------------------------模型训练部分----------------------------------------------------------------------------
#隐含变量维数
K=40

#item和用户的偏置项
bi=np.zeros((n_items,1))
bu=np.zeros((n_users,1))

#item和用户的隐含向量
pu=np.zeros((n_users,K))
qi=np.zeros((n_items,K))

#隐含向量初始化
from numpy.random import random
for uid in range(n_users):
    #能把这个矩阵写出来，真的是高手啊
    #random((K,1))  随机生成一个0-1之间的Kx1矩阵
    #reshape(random((K,1))/10*(np.sqrt(K)),K) 将生成的K行1列矩阵，变换成一个含有10个元素的列表
    pu[uid]=np.reshape(random((K,1))/10*(np.sqrt(K)),K)
for pid in range(n_items):
    qi[pid]=np.reshape(random((K,1))/10*(np.sqrt(K)),K)


#-------------------------------------训练速度较慢，采取分步计算，此步可省略----------------------------------------
#计算所有用户的平均打分
mu=df_train['fractional_rating_count'].mean()

#加载模型
# 用户和item的索引
users_index = pickle.load(open("/data/weixin-38664232/my-dataset/users_index.pkl", 'rb'))
items_index = pickle.load(open("/data/weixin-38664232/my-dataset/items_index.pkl", 'rb'))

n_users = len(users_index)
n_items = len(items_index)

# 用户-物品关系矩阵R
user_item_scores = sio.mmread("/data/weixin-38664232/my-dataset/user_item_scores")

# 倒排表
##每个用户播放的歌曲
user_items = pickle.load(open("/data/weixin-38664232/my-dataset/user_items.pkl", 'rb'))
##事件参加的用户
item_users = pickle.load(open("/data/weixin-38664232/my-dataset/item_users.pkl", 'rb'))


#-------------------------------------模型预测--------------------------------------------------------------------------
def svc_pred(uid,iid):
    score=mu+bi[iid]+bu[uid]+np.sum(qi[iid]*pu[uid])

    return score

#50遍训练，得到模型
steps=50
gamma=0.04
Lambda=0.15
n_records=df_train.shape[0]
for step in range(steps):
    print('the '+str(step)+' is running')

    #将训练样本打乱
    kk=np.random.permutation(n_records)
    #每次随机训练一个样本
    for j in range(n_records):
        line=kk[j] #kk[j]为输入j，得到j对应的行索引，因为模型训练过程中是以用户id所在的索引，所以这里也要找出对应的index
        uid=users_index[df_train.iloc[line]['msno']]
        iid=items_index[df_train.iloc[line]['song_id']]

        rating=df_train.iloc[line]['fractional_rating_count']

        #预测残差
        eui=rating-svc_pred(uid,iid)

        #随机梯度下降更新
        bu[uid]+=gamma*(eui-Lambda*bu[uid])
        bi[iid]+=gamma*(eui-Lambda*bi[iid])

        temp=qi[iid]
        qi[iid]+=gamma*(eui*pu[uid]-Lambda*qi[iid])
        pu[uid]+=gamma*(eui*temp-Lambda*pu[uid])

    #学习率递减
    gamma=gamma*0.93

#保存bu bi pu pi
import json
def save_json(filepath):
    dict_={}
    dict_['mu']=mu
    dict_['K']=K

    dict_['bu']=bu
    dict_['bi']=bi

    dict_['pu']=pu
    dict_['qi']=qi
    jsonfile=json.dumps(dict_)
    with open(filepath,'w') as file:
        file.write(jsonfile)

filepath='save_model.json'
save_json(filepath)


