import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
df_train=pd.read_csv('train.csv')
df_songs=pd.read_csv('songs.csv')
df_songs_extra=pd.read_csv('song_extra_info.csv')
df_members=pd.read_csv('members.csv')

df_test=pd.read_csv('test.csv')

print('训练集和测试集中 共有的用户：',len(set.intersection(
    set(df_train['msno']),set(df_test['msno'])
)))

print('训练集合测试集中都出现的歌曲：',len(set.intersection(
    set(df_train['song_id']),set(df_test['song_id'])
)))

print('训练集中的歌曲数量：',df_train['song_id'].unique().shape[0])
print('测试集中的歌曲数量：',df_test['song_id'].unique().shape[0])

print('训练集中的用户数量：',df_train['msno'].unique().shape[0])
print('测试集中的用户数量：',df_test['msno'].unique().shape[0])

plt.figure(figsize=(12,8))
sns.countplot(df_train['target'])
#plt.show()

#print(df_songs.head())

#将歌曲合并到训练集  合并原则:song_id
df_train=df_train.merge(df_songs,on='song_id',how='left')
#将歌曲额外信息合并到训练集 合并原则：song_id
df_train=df_train.merge(df_songs_extra,on='song_id',how='left')
print(df_train.head())

#条状图展示事件触发类型
plt.figure(figsize=(12,10))
sns.countplot(df_train['source_system_tab'],hue=df_train['target'])
#plt.show()

#查看 音乐入口类型
plt.figure(figsize=(12,10))
g=sns.countplot(df_train['source_type'],hue=df_train['target'])
locs,labels=plt.xticks()
g.set_xticklabels(labels,rotation=45)
#plt.show()

#去掉训练集中，song_length为缺失值的项
df_train.dropna(subset=['song_length'],inplace=True)
df_train.dropna(subset=['language'],inplace=True)

#将触发事件、播放音乐入口的类型改为category
df_train['source_system_tab']=df_train['source_system_tab'].astype('category')
df_train['source_type']=df_train['source_type'].astype('category')

#查看语言类型
print(df_train['language'].value_counts())

plt.figure(figsize=(12,10))
sns.countplot(df_train['language'],hue=df_train['target'])
#plt.show()

#从语言类型的柱状图中看出，3.0 52.0 31.0占了所有用户的一半

#将用户的信息合并到训练集
df_train=df_train.merge(df_members,on='msno',how='left')

plt.figure(figsize=(14,12))
df_train['bd'].value_counts().plot.bar()
plt.xlim([-1,100])
#plt.show()

#从柱状图中可以看出大约40%的用户年龄为0，
#查看年龄小于0的用户
print('年龄小于0的样本',len(df_train.query('bd<0')))

#去掉年龄小于0的样本
df_train=df_train.query('bd>=0')#195

#年龄大于100的样本
print('年领大于100的样本有多少：',len(df_train.query('bd>100')))#6508
#从年龄小于0 和大于100的数量来看，所占比例较小，因此删除年龄小于0，大于100的样本
df_train=df_train.query('bd >0 and bd <=80')

#查看训练集总体情况
print(df_train.info())


