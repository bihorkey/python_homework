import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
import os
import random
from chardet import detect  # 用于自动检测编码

def crawl_dalian_weather(year_month):
    """
    爬取大连市指定年月的每日天气数据（编码问题完全解决版）
    """
    url = f"https://www.tianqihoubao.com/lishi/dalian/month/{year_month}.html"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://www.tianqihoubao.com/'
    }
    
    try:
        print(f"正在爬取 {year_month}...")
        response = requests.get(url, headers=headers, timeout=10)
        
        # 自动检测网页编码
        encoding = detect(response.content)['encoding']
        response.encoding = encoding if encoding else 'gbk'
        print(f"检测到编码: {response.encoding}")
        
        if response.status_code != 200:
            print(f"请求失败，状态码: {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='weather-table') or soup.find('table', class_='b')
        if not table:
            print(f"未找到表格数据: {year_month}")
            return None
            
        data = []
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 4:
                continue
                
            # 提取日期
            date_link = cols[0].find('a')
            date_text = date_link.get_text(strip=True) if date_link else cols[0].get_text(strip=True)
            
            # 提取天气信息（不做任何清理）
            weather_text = cols[1].get_text(' ', strip=True).replace('\n', '')
            weather_parts = [w.strip() for w in weather_text.split('/')]
            day_weather = weather_parts[0] if weather_parts else ''
            night_weather = weather_parts[1] if len(weather_parts) > 1 else ''
            
            # 提取温度
            temp_text = cols[2].get_text(' ', strip=True)
            temp_numbers = re.findall(r'-?\d+', temp_text)
            high_temp = int(temp_numbers[0]) if temp_numbers and len(temp_numbers) >= 1 else np.nan
            low_temp = int(temp_numbers[1]) if temp_numbers and len(temp_numbers) >= 2 else np.nan
            
            # 提取风力信息（不做任何清理）
            wind_text = cols[3].get_text(' ', strip=True).replace('\n', '')
            wind_parts = [w.strip() for w in wind_text.split('/')]
            day_wind = wind_parts[0] if wind_parts else ''
            night_wind = wind_parts[1] if len(wind_parts) > 1 else ''
            
            data.append([
                date_text,
                day_weather,
                night_weather,
                high_temp,
                low_temp,
                day_wind,
                night_wind
            ])
        
        df = pd.DataFrame(data, columns=['日期', '白天天气', '夜晚天气', '最高温', '最低温', '白天风力', '夜晚风力'])
        print(f"成功爬取 {year_month}，共 {len(df)} 条记录")
        
        # 打印前几行数据以便检查
        print("样例数据:")
        print(df.head(2).to_string(index=False))
        
        return df
    
    except Exception as e:
        print(f"爬取 {year_month} 失败: {str(e)}")
        return None

# 在crawl_full_period()函数中添加Excel导出
def crawl_full_period():
    all_data = []
    failed_months = []
    
    for year in [2022, 2023, 2024]:
        for month in range(1, 13):
            year_month = f"{year}{month:02d}"
            df_month = crawl_dalian_weather(year_month)
            if df_month is not None:
                all_data.append(df_month)
            else:
                failed_months.append(year_month)
            # time.sleep(random.uniform(1, 3))
    
    if all_data:
        df_full = pd.concat(all_data, ignore_index=True)
        # df_full['日期'] = pd.to_datetime(df_full['日期'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        output_dir = "homework/second/data"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV（GBK编码）
        csv_filename = os.path.join(output_dir, "dalian_weather_2022-2024.csv")
        df_full.to_csv(csv_filename, index=False, encoding='gbk', errors='ignore')
        
        
        print(f"CSV文件已保存至: {csv_filename}")
        return df_full
    else:
        print("爬取失败，无有效数据")
        return None

if __name__ == "__main__":
    print("大连市2022-2024年天气数据爬取程序")
    weather_data = crawl_full_period()