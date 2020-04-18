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
    with open('cut.txt','w',encoding='UTF-8') as f:
        for line in df['content']:
            seg_line = cutTxt(line)
            f.writelines(seg_line+ '\n')