import pandas as pd
import jieba

def cutTxt(line):
    cutList = jieba.cut(line,cut_all = False)
    cutSent = ""
    for temp in cutList:
        if temp != '\t':
            cutSent += temp + ' '
    return cutSent.strip()

