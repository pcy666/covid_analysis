import pandas as pd
import jieba

def cutTxt(line):
    cutList = jieba.cut(line,cut_all = False)
    cutSent = ""
    for temp in cutList:
        if temp != '\t':
            cutSent += temp + ' '
    return cutSent.strip()

if __name__ == '__main__':
    df = pd.read_csv('Information.csv')
    f = open('cut.txt','w',encoding='UTF-8')

    for line in df['content']:
        seg_line = cutTxt(line)
        f.writelines(seg_line + '\n')
    f.close()