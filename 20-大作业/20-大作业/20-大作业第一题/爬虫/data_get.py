import requests
import pandas as pd
import random
from time import sleep
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


data_list = []

# 随机User-Agent列表
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
]


def fetch_page(page_num, total_pages):
    base_url = 'https://www.hurun.net/zh-CN/Rank/HsRankDetailsList'
    params = {
        'num': 'ODBYW2BI',
        'search': '',
        'offset': (page_num - 1) * 20,
        'limit': 20
    }

    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Referer': 'https://www.hurun.net/zh-CN/Rank/HsRankDetails?pagetype=rich',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://www.hurun.net'
    }

    try:
        sleep_time = random.uniform(0.2, 0.5)
        print(f"等待{sleep_time:.2f}秒后爬取第{page_num}/{total_pages}页")
        sleep(sleep_time)

        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        page_data = response.json()
        rows = page_data.get('rows', [])

        if not rows:
            print(f"第{page_num}页无数据，跳过")
            return []

        page_items = []
        for item in rows:
            data = {
                '排名': item.get('hs_Rank_Rich_Ranking', ''),
                '排名变化': item.get('hs_Rank_Rich_Ranking_Change', ''),
                '姓名_中文': '',
                '姓名_英文': '',
                '年龄': '',
                '性别': '',
                '出生地_中文': '',
                '出生地_英文': '',
                '常住地_中文': '',
                '教育程度': '',
                '毕业院校': '',  # 新增字段：确保正确提取
                '公司名称_中文': item.get('hs_Rank_Rich_ComName_Cn', ''),
                '公司名称_英文': item.get('hs_Rank_Rich_ComName_En', ''),
                '公司总部_中文': item.get('hs_Rank_Rich_ComHeadquarters_Cn', ''),
                '公司总部_英文': item.get('hs_Rank_Rich_ComHeadquarters_En', ''),
                '所在行业_中文': item.get('hs_Rank_Rich_Industry_Cn', ''),
                '所在行业_英文': item.get('hs_Rank_Rich_Industry_En', ''),
                '财富值_人民币_亿': item.get('hs_Rank_Rich_Wealth', ''),
                '财富值_美元_万': item.get('hs_Rank_Rich_Wealth_USD', ''),
                '财富变化': item.get('hs_Rank_Rich_Wealth_Change', ''),
                '财富来源': item.get('hs_Rank_Rich_Wealth_Source_Cn', ''),
                '关系': item.get('hs_Rank_Rich_Relations', ''),
                '年份': item.get('hs_Rank_Rich_Year', '')
            }

            # 提取人物详情
            if 'hs_Character' in item and isinstance(item['hs_Character'], list) and len(item['hs_Character']) > 0:
                character = item['hs_Character'][0]  # 取第一个人物信息

                # 填充个人信息字段
                data['姓名_中文'] = character.get('hs_Character_Fullname_Cn', data['姓名_中文'])
                data['姓名_英文'] = character.get('hs_Character_Fullname_En', data['姓名_英文'])
                data['年龄'] = character.get('hs_Character_Age', data['年龄'])
                data['性别'] = character.get('hs_Character_Gender', data['性别'])
                data['出生地_中文'] = character.get('hs_Character_BirthPlace_Cn', data['出生地_中文'])
                data['出生地_英文'] = character.get('hs_Character_BirthPlace_En', data['出生地_英文'])
                data['常住地_中文'] = character.get('hs_Character_Permanent_Cn', data['常住地_中文'])
                data['教育程度'] = character.get('hs_Character_Education_Cn', data['教育程度'])
                data['毕业院校'] = character.get('hs_Character_School_Cn', '')

            page_items.append(data)

        print(f"第{page_num}页爬取完成，获取{len(page_items)}条数据")
        return page_items

    except Exception as e:
        print(f"第{page_num}页失败：{str(e)}")
        return []


def main():
    try:
        test_url = 'https://www.hurun.net/zh-CN/Rank/HsRankDetailsList'
        test_params = {'num': 'ODBYW2BI', 'offset': 0, 'limit': 20}
        test_headers = {'User-Agent': random.choice(user_agents)}

        response = requests.get(test_url, params=test_params, headers=test_headers)
        response.raise_for_status()
        total_data = response.json()
        total = total_data.get('total', 0)
        total_pages = (total + 19) // 20

        print(f"总数据量：{total}条，共{total_pages}页，开始爬取...")

        # 并发爬取
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_page, p, total_pages): p for p in range(1, total_pages + 1)}

            for future in as_completed(futures):
                page = futures[future]
                try:
                    page_data = future.result()
                    data_list.extend(page_data)
                except Exception as e:
                    print(f"处理第{page}页结果时出错：{str(e)}")

        # 保存并验证数据完整性
        if data_list:
            df = pd.DataFrame(data_list)

            # 按排名排序
            df['排名'] = pd.to_numeric(df['排名'], errors='coerce')
            df = df.sort_values('排名').reset_index(drop=True)

            # 保存完整数据（
            df.to_csv('2024胡润百富榜.csv', index=False, encoding='utf_8_sig')

            # 验证结果：显示字段数量和部分数据
            print(f"\n爬取完成！共{len(df)}条数据，包含{len(df.columns)}个字段")
            print("字段列表：", ', '.join(df.columns.tolist()))  # 确认所有字段都在
            print("\n前5条数据预览（含毕业院校）：")
            print(df[['排名', '姓名_中文', '毕业院校', '公司名称_中文', '财富值_人民币_亿']].head().to_string(
                index=False))
        else:
            print("未获取到任何数据")

    except Exception as e:
        print(f"爬取失败：{str(e)}")


if __name__ == "__main__":
    main()