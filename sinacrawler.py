import requests
import json
import csv
from urllib.parse import urlencode
from pyquery import PyQuery as pq
import os
import time
import re

headers = {
    'Host': 'm.weibo.cn',
    'Referer': 'https://m.weibo.cn/u/2830678474',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}

path = 'Information.csv'

def GetJson(title,page):
    url = r'https://m.weibo.cn/api/container/getIndex?'
    param = {
        'containerid': '100103type=1&q='+title,
        'pagetype':'searchall',
        'page':page #page是就是当前处于第几页，是我们要实现翻页必须修改的内容。
    }
    try:
        response = requests.get((url+urlencode(param)), headers=headers)
        if response.status_code == 200:
            print('Get %s page successfully'%page)
            return response.json()
    except BaseException:
        print('Error')

def Clean(text:str):
    if(text != ''):
        text = text.strip()
        # 去除文本中的英文和数字
        text = re.sub("[a-zA-Z0-9]", "", text)
        # 去除文本中的中文符号和英文符号
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", text)
        return text
    return None

if not os.path.exists(path):
    with open(path, "w", newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["content","estimate"])

with open(path, "a", newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    for page in range(2,100):#循环页面
        time.sleep(1)         #设置睡眠时间，防止被封号
        cards = GetJson('新冠',page).get('data').get('cards')
        for i in cards:
            text = Clean(pq(i.get('mblog').get('text')).text().replace(" ", "").replace("\n" , ""))
            # text = pq(i.get('mblog').get('text')).text().replace(" ", "").replace("\n" , "")
            if text[-2:] == '全文':
                text = text[:-2]
            # text = text.encode('utf-8')
            writer.writerow([text])

