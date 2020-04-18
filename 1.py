import pandas as pd
import jieba

data = "你好吗我1一点都2不好我不喜欢吃水果"
output = jieba.cut(data,cut_all=False)
print(list(output))