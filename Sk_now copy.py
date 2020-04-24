import pandas as pd
import numpy as np
from math import log
from collections import Counter

data = pd.read_csv('cut.csv')
test = pd.read_csv('cut_test.csv')
data.head()

x = data['content']     #存储分词后的内容
y = data['estimate']    #存储用户情感倾向
x_test = test['content']    #存储测试样本的内容
y_test = test['estimate']   #存储测试样本的用户情感倾向

from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

from sklearn.feature_extraction.text import CountVectorizer

def txt2list(wordstxt):
    with open(wordstxt) as f:
        reader = f.read()
    stopwords_list = reader.split('\n')     #按行存储为list
    out_stopwords_list = [i for i in stopwords_list]
    return out_stopwords_list

filename = '停用词表.txt'
stopwords = txt2list(filename)

vect = CountVectorizer(max_df = 0.8, 
                       min_df = 3, 
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', 
                       stop_words=frozenset(stopwords))     #词频统计

_list = vect.fit_transform(x).toarray()     #训练样本词频统计
test_list = vect.transform(x_test).toarray()        #检验样本词频统计
a = pd.DataFrame(_list, columns=vect.get_feature_names())       #创建数据帧用于显示
a.head()

print(_list)
print(test_list)

y_array = np.array(y)       #训练样本情感倾向
y_test_array = np.array(y_test)     #检验样本情感倾向
print(y_array)
print(y_test_array)

y_static = Counter(y_array.flatten())   #统计情感倾向类型
sumlist = []
sumlist.append(y_static.get(0))
sumlist.append(y_static.get(1))
sumlist.append(y_static.get(2))
sum = sumlist[0]+sumlist[1]+sumlist[2]
print("the sum is : {}\nthe sum of 0 is :{}\nthe sum of 1 is :{}\nthe sum of 2 is :{}".format(sum,sumlist[0],sumlist[1],sumlist[2]))

prior = [i/sum for i in sumlist]    #每种倾向发生的次数/总次数
print(prior)

y_dic = {'0':[],'1':[],'2':[]} #字典初始化

rows,cols = _list.shape#计算训练样本维度

for i in range(rows):
    y_dic[str(y_array[i])].append(_list[i])     #统计每种倾向的词向量
for i in range(3):
    y_dic[str(i)] = np.array(y_dic[str(i)])     #格式转换
print(y_dic)

conp = {'0':[],'1':[],'2':[]}
summary_ = []
for i in range(3):
    summary_.append(np.cumsum(y_dic[str(i)],axis=0)[y_dic[str(i)].shape[0]-1])      #计算每个词向量的累加值
print(summary_)

wordsum = []
for i in range(3):
    wordsum.append(summary_[i].sum(axis = 0))
print(wordsum)

for i in range(3):
    conp[str(i)] = (summary_[i]+1)/(sumlist[i]+2) #拉普拉斯平滑处理（防止出现0概率对乘除计算的影响）
    for j in range(len(conp[str(i)])):
        conp[str(i)][j] = log(conp[str(i)][j])#转化为对数形式（和条件概率正相关）
print(conp)

y_results = []
for i in range(rows):
    p0 = p1 = p2 = 0
    for j in range(cols):
        p0 = p0 + ((conp['0'][j])*(_list[i][j]))    #条件概率*先验概率
        p1 = p1 + ((conp['1'][j])*(_list[i][j]))
        p2 = p2 + ((conp['2'][j])*(_list[i][j]))
    if(p0 > p1 and p0 > p2):y_results.append(0)
    elif(p1 > p0 and p1 > p2):y_results.append(1)
    elif(p2 > p0 and p2 > p1):y_results.append(2)
    else:y_results.append(0)
score = 1-(np.int64((y_results-y_array) != 0).sum())/(np.int64((y_results-y_array) != 0).shape[0])
print(score)

y_test_results = []
test_rows, a = test_list.shape #只用到test_rows变量
print(test_rows)
for i in range(test_rows):
    p0 = p1 = p2 = 0
    for j in range(cols):
        p0 = p0 + ((conp['0'][j])*(test_list[i][j]))
        p1 = p1 + ((conp['1'][j])*(test_list[i][j]))
        p2 = p2 + ((conp['2'][j])*(test_list[i][j]))
    if(p0 > p1 and p0 > p2):y_test_results.append(0)
    elif(p1 > p0 and p1 > p2):y_test_results.append(1)
    elif(p2 > p0 and p2 > p1):y_test_results.append(2)
    else:y_test_results.append(0)
test_score = 1-(np.int64((y_test_results-y_test_array) != 0).sum())/(np.int64((y_test_results-y_test_array) != 0).shape[0])
print(test_score)