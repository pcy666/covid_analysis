import pandas as pd
import jieba

data = "你好吗我一点都不好我不喜欢吃水果"
output = jieba.cut(data,cut_all=False)
print(list(output))