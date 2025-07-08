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
# print(data[['日期', '年份', '月份']].head())

# print(data)
# 绘制最高温度和最低温度的折线图
def plot_temperature(data):
    plt.figure(figsize=(12, 6), dpi=100)
    plt.title('大连市2022-2024年月平均温度变化趋势', fontsize=14, pad=20)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('温度 (℃)', fontsize=12)
    
    monthly_high = data.groupby('月份')['最高温'].mean()
    monthly_low = data.groupby('月份')['最低温'].mean()
    
    plt.plot(
        monthly_high.index, monthly_high.values, 
        color='#E74C3C', linewidth=2, marker='o', 
        markersize=8, label='最高气温', alpha=0.8
    )
    plt.plot(
        monthly_low.index, monthly_low.values, 
        color='#3498DB', linewidth=2, linestyle='--', 
        marker='s', markersize=8, label='最低气温', alpha=0.8
    )
    
    # 标注极值点
    max_temp = monthly_high.max()
    min_temp = monthly_low.min()
    plt.annotate(f'最高: {max_temp:.1f}℃', 
                 xy=(monthly_high.idxmax(), max_temp),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'最低: {min_temp:.1f}℃', 
                 xy=(monthly_low.idxmin(), min_temp),
                 xytext=(10, -20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))
    
    # 美化坐标轴和网格
    plt.xticks(range(1, 13), [f'{m}月' for m in range(1, 13)], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', fontsize=12)
    
    # 调整边距
    plt.tight_layout()
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

def plot_wind_distribution(data, year_range="2022-2024"):
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取风力等级
    data['白天风力等级'] = data['白天风力'].apply(wind_level)
    data['夜间风力等级'] = data['夜晚风力'].apply(wind_level)
    
    # 定义风力等级顺序和配色
    wind_order = ['1-2级', '3-4级', '5-6级', '7级以上']
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(wind_order)))
    
    # 统计风力天数
    def get_wind_stats(df, col):
        counts = df.groupby(['月份', col]).size().unstack(fill_value=0)
        for level in wind_order:
            if level not in counts.columns:
                counts[level] = 0
        return counts[wind_order]
    
    day_counts = get_wind_stats(data, '白天风力等级')
    night_counts = get_wind_stats(data, '夜间风力等级')
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12), dpi=100)
    fig.suptitle(f'大连市{year_range}年风力等级分布（按月份统计）', 
                fontsize=18, y=1.02)
    
    # 公用参数
    bar_width = 0.2
    months = np.arange(1, 13)
    y_max = max(day_counts.max().max(), night_counts.max().max()) + 3
    
    # 绘制白天风力分布
    ax1 = plt.subplot(2, 1, 1)
    for i, (level, color) in enumerate(zip(wind_order, colors)):
        pos = months + i * bar_width
        bars = ax1.bar(pos, day_counts[level], width=bar_width,
                      color=color, label=level, alpha=0.8)
        # 智能数据标签位置
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height+0.3,
                        f'{height}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_title('白天风力分布', fontsize=14, pad=10)
    ax1.set_ylabel('出现天数', fontsize=12)
    ax1.set_xticks(months + bar_width*1.5)
    ax1.set_xticklabels([f'{m}月' for m in months], fontsize=10)
    ax1.set_ylim(0, y_max)
    ax1.legend(title='风力等级', bbox_to_anchor=(1.02, 1), 
              borderaxespad=0)
    ax1.grid(axis='y', linestyle=':', alpha=0.4)
    
    # 绘制夜间风力分布
    ax2 = plt.subplot(2, 1, 2)
    for i, (level, color) in enumerate(zip(wind_order, colors)):
        pos = months + i * bar_width
        bars = ax2.bar(pos, night_counts[level], width=bar_width,
                      color=color, label=level, alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height+0.3,
                        f'{height}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_title('夜间风力分布', fontsize=14, pad=10)
    ax2.set_xlabel('月份', fontsize=12)
    ax2.set_ylabel('出现天数', fontsize=12)
    ax2.set_xticks(months + bar_width*1.5)
    ax2.set_xticklabels([f'{m}月' for m in months], fontsize=10)
    ax2.set_ylim(0, y_max)
    ax2.grid(axis='y', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    
    plt.show()
    
    return day_counts, night_counts

# 天气类型标准化函数
def standardize_weather(weather_str):
    weather_str = str(weather_str).strip()
    if('雨夹雪') in weather_str or '雨雪' in weather_str:
        return '雪天'
    elif('雷') in weather_str:
        return '雷雨天'
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

weather_order = ['晴天', '多云', '阴天', '雨天', '雪天', '雷雨天']
colors = ['#FFD700', '#87CEEB', '#A9A9A9', '#4682B4', '#B0E0E6', '#9370DB']

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
plot_wind_distribution(data)
# 调用函数绘制天气分布图
bar_weather_distribution(data)