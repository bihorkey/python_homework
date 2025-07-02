import requests
import pandas as pd
import random
from time import sleep

# 分页爬取15页（每页200条）
data_list = []
for page in range(1, 16):
    offset = (page - 1) * 200
    url = f'https://www.hurun.net/zh-CN/Rank/HsRankDetailsList?num=ODBYW2BI&offset={offset}&limit=200'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Referer': 'https://www.hurun.net/zh-CN/Rank/HsRankDetails?pagetype=rich'
    }
    sleep(random.uniform(1, 2))  # 反爬延迟
    response = requests.get(url, headers=headers)
    json_data = response.json()['data']
    data_list.extend(json_data)

# 数据解析与DataFrame构建
df = pd.DataFrame(data_list)
df.to_csv('hurun_2024.csv', encoding='utf_8_sig', index=False)  # 解决中文乱码