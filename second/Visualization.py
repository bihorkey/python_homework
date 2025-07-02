import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

data = pd.read_csv('homework/second/data/dalian_weather_2022-2024.csv', encoding='gbk')   
# 将日期列转换为datetime类型
data['日期'] = pd.to_datetime(data['日期'], format='%Y年%m月%d日')

# 从日期中提取年份和月份
data['年份'] = data['日期'].dt.year
data['月份'] = data['日期'].dt.month

# 确认转换是否正确
print(data[['日期', '年份', '月份']].head())

print(data)
# 绘制最高温度和最低温度的折线图
def plot_temperature(data):
    plt.figure(figsize=(12, 6))
    plt.title('大连市2022-2024年温度变化趋势')
    plt.xlabel('月份')
    plt.ylabel('温度 (℃)')
    
    # 准备x轴数据（1-12月）
    months = range(1, 13)
    
    # 为每一年绘制折线
    # 计算每月的平均最高温度
    monthly_high_avg = data.groupby('月份')['最高温'].mean()
    monthly_low_avg = data.groupby('月份')['最低温'].mean()
    # 绘制折线图
    plt.plot(months, monthly_high_avg, label=f'最高气温', marker='o')
    plt.plot(months, monthly_low_avg, linestyle='--',label=f'最低气温', marker='x')
    # 添加图例和网格
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(months)
    plt.show()
        
        

# 风力等级划分函数
def wind_level(wind_str):
    wind_str = str(wind_str)  # 确保是字符串类型
    if '1-2' in wind_str or '1级' in wind_str or '2级' in wind_str:
        return '1-2级'
    elif '3-4' in wind_str or '3级' in wind_str or '4级' in wind_str:
        return '3-4级'
    elif '5-6' in wind_str or '5级' in wind_str or '6级' in wind_str:
        return '5-6级'
    else:
        return '7级以上'

# 风力柱状图绘制函数
def bar_wind_distribution(data):
    # 提取白天和夜间风力等级
    data['白天风力等级'] = data['白天风力'].apply(wind_level)
    data['夜间风力等级'] = data['夜晚风力'].apply(wind_level)
    
    # 按月份和风力等级分组统计天数（白天）
    day_wind_counts = data.groupby(['月份', '白天风力等级']).size().unstack(fill_value=0)
    
    # 按月份和风力等级分组统计天数（夜间）
    night_wind_counts = data.groupby(['月份', '夜间风力等级']).size().unstack(fill_value=0)
    
    # 定义统一的风力等级顺序
    wind_order = ['1-2级', '3-4级', '5-6级', '7级以上']
    
    # 确保两个统计表有相同的列顺序
    for level in wind_order:
        if level not in day_wind_counts.columns:
            day_wind_counts[level] = 0
        if level not in night_wind_counts.columns:
            night_wind_counts[level] = 0
    
    day_wind_counts = day_wind_counts[wind_order]
    night_wind_counts = night_wind_counts[wind_order]
    
    # 创建图形和子图
    plt.figure(figsize=(16, 10))
    
    # 设置柱状图位置和宽度
    bar_width = 0.2
    months = np.arange(1, 13)
    
    # ========== 白天风力分布子图 ==========
    plt.subplot(2, 1, 1)  # 2行1列，第一个子图
    
    # 为每个风力等级绘制柱状图
    for i, level in enumerate(wind_order):
        # 计算每个柱子的位置
        positions = months + i * bar_width
        # 绘制柱状图
        plt.bar(positions, day_wind_counts[level], width=bar_width, 
                label=f'{level}', alpha=0.8)
    
    # 添加标题和标签
    plt.title('大连市近三年白天风力等级分布', fontsize=16)
    plt.ylabel('出现天数', fontsize=12)
    
    # 设置x轴刻度和标签
    plt.xticks(months + bar_width * 1.5, [f'{m}月' for m in months])
    max_day = day_wind_counts.max().max()
    plt.ylim(0, max_day + 5)  # 设置y轴范围
    
    # 添加图例和网格
    plt.legend(title='风力等级', fontsize=10, loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for i, level in enumerate(wind_order):
        positions = months + i * bar_width
        for j, pos in enumerate(positions):
            count = day_wind_counts[level].iloc[j]
            if count > 0:  # 只显示非零值
                plt.text(pos, count + 0.5, str(count), 
                         ha='center', va='bottom', fontsize=9)
    
    # ========== 夜间风力分布子图 ==========
    plt.subplot(2, 1, 2)  # 2行1列，第二个子图
    
    # 为每个风力等级绘制柱状图
    for i, level in enumerate(wind_order):
        # 计算每个柱子的位置
        positions = months + i * bar_width
        # 绘制柱状图
        plt.bar(positions, night_wind_counts[level], width=bar_width, 
                label=f'{level}', alpha=0.8)
    
    # 添加标题和标签
    plt.title('大连市近三年夜间风力等级分布', fontsize=16)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('出现天数', fontsize=12)
    
    # 设置x轴刻度和标签
    plt.xticks(months + bar_width * 1.5, [f'{m}月' for m in months])
    max_night = night_wind_counts.max().max()
    plt.ylim(0, max_night + 5)  # 设置y轴范围
    
    # 添加网格
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for i, level in enumerate(wind_order):
        positions = months + i * bar_width
        for j, pos in enumerate(positions):
            count = night_wind_counts[level].iloc[j]
            if count > 0:  # 只显示非零值
                plt.text(pos, count + 0.5, str(count), 
                         ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # 返回统计结果供参考
    return day_wind_counts, night_wind_counts

# 天气类型标准化函数
def standardize_weather(weather_str):
    weather_str = str(weather_str).strip()
    if('雨夹雪') in weather_str or '雨雪' in weather_str:
        return '雪天'
    elif '晴' in weather_str:
        return '晴天'
    elif '多云' in weather_str:
        return '多云'
    elif '阴' in weather_str:
        return '阴天'
    elif '雨' in weather_str:
        return '雨天'
    elif '雪' in weather_str:
        return '雪天'

# 处理白天和夜晚天气数据
data['白天天气'] = data['白天天气'].apply(standardize_weather)
data['夜间天气'] = data['夜晚天气'].apply(standardize_weather)

# 定义天气类型顺序
weather_order = ['晴天', '多云', '阴天', '雨天', '雪天']

def bar_weather_distribution(data):
    # 统计白天和夜晚天气分布
    day_weather = data.groupby(['月份', '白天天气']).size().unstack(fill_value=0)
    night_weather = data.groupby(['月份', '夜间天气']).size().unstack(fill_value=0)
    
    # 确保所有天气类型都存在
    for w in weather_order:
        if w not in day_weather.columns:
            day_weather[w] = 0
        if w not in night_weather.columns:
            night_weather[w] = 0
    
    day_weather = day_weather[weather_order]
    night_weather = night_weather[weather_order]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 设置柱状图参数
    bar_width = 0.12
    months = np.arange(1, 13)
    colors = plt.cm.tab20(np.linspace(0, 1, len(weather_order)))
    
    # 绘制白天天气分布
    for i, weather in enumerate(weather_order):
        ax1.bar(months + i*bar_width, day_weather[weather], 
                width=bar_width, color=colors[i], label=weather)
    
    ax1.set_title('大连市近三年白天天气分布', fontsize=16)
    ax1.set_ylabel('天数', fontsize=12)
    ax1.set_xticks(months + bar_width*3)
    ax1.set_xticklabels([f'{m}月' for m in months])
    ax1.legend(bbox_to_anchor=(1, 1))
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 绘制夜间天气分布
    for i, weather in enumerate(weather_order):
        ax2.bar(months + i*bar_width, night_weather[weather], 
                width=bar_width, color=colors[i], label=weather)
    
    ax2.set_title('大连市近三年夜间天气分布', fontsize=16)
    ax2.set_xlabel('月份', fontsize=12)
    ax2.set_ylabel('天数', fontsize=12)
    ax2.set_xticks(months + bar_width*3)
    ax2.set_xticklabels([f'{m}月' for m in months])
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    return day_weather, night_weather


# 调用函数绘制温度变化趋势图
plot_temperature(data)
# 调用函数绘制风力分布图
bar_wind_distribution(data)
# 调用函数绘制天气分布图
bar_weather_distribution(data)

