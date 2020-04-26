import pandas as pd
import numpy as np
from math import log
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from os import path
import matplotlib.pyplot as plt

def txt2list(wordstxt):#读取停用词表
    with open(wordstxt) as f:
        reader = f.read()
    stopwords_list = reader.split('\n')     #按行存储为list
    out_stopwords_list = [i for i in stopwords_list]
    return out_stopwords_list



def VectInit(train,vect):#词向量模型初始化函数
    _list = vect.fit_transform(train).toarray()     #获得训练样本的词向量
    return _list    #返回拟合训练样本后的模型



def PreNB(_list,y):     #朴素贝叶斯算法预处理,使用训练样本计算条件概率和先验概率并返回
    y_array = np.array(y)       #训练样本情感倾向
    y_static = Counter(y_array.flatten())   #统计情感倾向类型
    sumlist = []
    sumlist.append(y_static.get(0))
    sumlist.append(y_static.get(1))
    sumlist.append(y_static.get(2))
    sum = sumlist[0]+sumlist[1]+sumlist[2]

    prior = [i/sum for i in sumlist]    #得到先验概率

    y_dic = {'0':[],'1':[],'2':[]} #字典初始化
    (rows,cols) = _list.shape#计算训练样本维度
    for i in range(rows):
        y_dic[str(y_array[i])].append(_list[i])     #将各个训练数据放入对应分类的字典中
    for i in range(3):
        y_dic[str(i)] = np.array(y_dic[str(i)])     #格式转换为array格式用于后面累加

    summary_ = []
    for i in range(3):
        summary_.append(np.cumsum(y_dic[str(i)],axis=0)[y_dic[str(i)].shape[0]-1])      #计算每个词向量的累加值

    wordsum = []
    for i in range(3):
        wordsum.append(summary_[i].sum(axis = 0))

    conp = {'0':[],'1':[],'2':[]}
    for i in range(3):
        conp[str(i)] = (summary_[i]+1)/(sumlist[i]+2) #拉普拉斯平滑处理（防止出现0概率对乘除计算的影响）
        for j in range(len(conp[str(i)])):
            conp[str(i)][j] = conp[str(i)][j]       
    return (prior,conp)
    


def NB(test_list,y,prior,conp):#测试函数,读取测试样本并返回准确率
    # print(prior)
    y_test_results = []
    y_test_array = np.array(y)     #检验样本情感倾向
    (rows,cols) = test_list.shape #只用到test_rows变量
    # print(rows,cols)
    for i in range(rows):
        p0 = p1 = p2 = 1
        for j in range(cols):
            p0 = p0 * ((conp['0'][j])**(test_list[i][j]))
            p1 = p1 * ((conp['1'][j])**(test_list[i][j]))
            p2 = p2 * ((conp['2'][j])**(test_list[i][j]))
        p0 = p0*prior[0]
        p1 = p1*prior[1]
        p2 = p2*prior[2]
        if(p0 > p1 and p0 > p2):y_test_results.append(0)
        elif(p1 > p0 and p1 > p2):y_test_results.append(1)
        elif(p2 > p0 and p2 > p1):y_test_results.append(2)
        else:y_test_results.append(0)
    test_score = 1-(np.int64((y_test_results-y_test_array) != 0).sum())/(np.int64((y_test_results-y_test_array) != 0).shape[0])
    return test_score



def NBforecast(forecast_list,prior,conp):#预测函数,返回预测数据的正面舆论率
    results = []
    (rows,cols) = forecast_list.shape #只用到test_rows变量
    # print(rows,cols)
    for i in range(rows):
        p0 = p1 = p2 = 1
        for j in range(cols):
            p0 = p0 * ((conp['0'][j])**(forecast_list[i][j]))
            p1 = p1 * ((conp['1'][j])**(forecast_list[i][j]))
            p2 = p2 * ((conp['2'][j])**(forecast_list[i][j]))
        p0 = p0*prior[0]
        p1 = p1*prior[1]
        p2 = p2*prior[2]
        if(p0 > p1 and p0 > p2):results.append(0)
        elif(p1 > p0 and p1 > p2):results.append(1)
        elif(p2 > p0 and p2 > p1):results.append(2)
        else:results.append(0)
    # print(results)
    results = np.array(results)
    rate = (np.sum(results == 1))/(np.sum(results==0)+np.sum(results!=0)) #计算正面舆论占比
    return rate

if __name__ == "__main__":
    filename = '停用词表.txt'
    stopwords = txt2list(filename)

    data = pd.read_csv('train.csv')
    test = pd.read_csv('cut_test.csv')
    x = data['content']     #存储分词后的内容
    y = data['estimate']    #存储用户情感倾向
    x_test = test['content']    #存储测试样本的内容
    y_test = test['estimate']   #存储测试样本的用户情感倾向

    vect = CountVectorizer(max_df = 0.8, min_df = 3, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', stop_words=frozenset(stopwords))     #词频统计模型
    _list = VectInit(x,vect)
    (prior,conp) = PreNB(_list,y)  #使用训练样本训练出先验概率和条件概率返回
    # print(conp)
    test_list = vect.transform(x_test).toarray()
    print('测试结果为:{}'.format(NB(test_list,y_test,prior,conp)))

    datadic = {'1':'cut_20200421.csv','2':'cut_20200422.csv','3':'cut_20200423.csv','4':'cut_20200424.csv','5':'cut_20200425.csv','6':'cut_20200426.csv'}
    rates = []
    for i in range(len(datadic)):#6个文件
        dataname = path.join('data',datadic[str(i+1)])#路径
        forecast_data = pd.read_csv(dataname)##打开文件
        x_forecast = forecast_data['content']#读取文本内容
        x_forecast = vect.transform(x_forecast).toarray()#获取词向量
        rates.append(NBforecast(x_forecast,prior,conp))#各个文件预测后的正面舆论率放入列表
    # print(rates)
    x = [21,22,23,24,25,26]
    plt.plot(x, rates)#画图
    plt.show()#显示图像