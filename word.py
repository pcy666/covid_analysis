import pandas as pd
import jieba
import csv

def cutTxt(line):
    cutList = jieba.cut(line,cut_all = False)
    cutSent = ""
    for temp in cutList:
        if temp != '\t':
            cutSent += temp + ' '
    return cutSent.strip()

if __name__ == '__main__':
    df = pd.read_csv('Information.csv')
    with open('cut.csv','w',encoding='UTF-8') as f:
        writer = csv.writer(f,delimiter = ',')
        writer.writerow(['content','estimate'])
        seg_line = ['','']
        for line in df.values:
            seg_line[0] = cutTxt(line[0])
            seg_line[1] = str(line[1])
            writer.writerow(seg_line)