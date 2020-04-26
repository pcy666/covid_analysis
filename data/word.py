import pandas as pd
import jieba
import csv

def cutTxt(line):
    cutList = jieba.cut(line,cut_all = False)
    cutSent = ""
    for temp in cutList:
        if temp != '\t'and temp != '【' and temp != '】':
            cutSent += temp + ' '
    return cutSent.strip()

def cutCsv(csv_name):#只对文本内容进行分析
    df = pd.read_csv(csv_name)
    with open("cut_{}".format(csv_name),'w',encoding='UTF-8') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(['content'])
        seg_line = ['']
        for line in df.values:
            seg_line[0] = cutTxt(line[0])
            writer.writerow(seg_line)

if __name__ == '__main__':
    cutCsv('20200426.csv')