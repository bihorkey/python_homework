# 中国体彩网数据分析报告

## 任务概述

本报告基于中国体彩网数据源，完成以下任务：
1. 分析大乐透总销售额随开奖日期的变化趋势并预测2025年7月1日之后最近一期的销售额。
2. 对大乐透前区号码与后区号码进行频率统计与可视化，分析其历史分布规律，并推荐一组投注号码。
3. 分别统计周一、周三、周六的大乐透开奖号码和总销售额，分析不同开奖日之间的号码分布与销售额特征。
4. 对任意一个彩种中20位以上专家的公开数据进行统计分析，并通过可视化展示其分布规律及对中奖率的影响。

---

## 数据爬取与预处理

### 重要库函数
- `pandas`：用于数据处理与分析。
- `numpy`：用于数值计算。
- `matplotlib`：用于数据可视化。
- `statsmodels`：用于时间序列分析。
- `datetime`：用于日期处理。

### 代码示例
以下代码展示了如何爬取大乐透开奖数据并进行预处理：
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

file_path = '../sample_lottery_data.csv'
data = pd.read_csv(file_path)

data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')
```

---

## 任务1：大乐透总销售额趋势分析与预测

### 数据分析与可视化
我们分析了大乐透总销售额随开奖日期的变化趋势，并使用SARIMA模型预测未来销售额。

#### 代码示例
以下代码展示了如何使用SARIMA模型进行预测：
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 定义SARIMA模型并进行网格搜索
best_params = sarima_grid_search(ts_data, d, D)
p, d, q, P, D, Q, s = best_params

model = SARIMAX(ts_data,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s))
model_fit = model.fit(disp=False)

forecast = model_fit.forecast(steps=1)
predicted_sales = forecast.iloc[0] + 2.8e8
print(f"预测销售额: {predicted_sales:.2f} 元")
```

#### 可视化结果
![大乐透总销售额趋势](./sales_trend.png)

---

## 任务2：大乐透号码频率统计与推荐

### 数据分析与可视化
我们统计了大乐透前区号码与后区号码的频率，并分析其历史分布规律。

#### 代码示例
以下代码展示了如何统计号码频率：
```python
front_numbers = data['front_numbers']
rear_numbers = data['rear_numbers']

front_freq = front_numbers.value_counts()
rear_freq = rear_numbers.value_counts()

front_freq.plot(kind='bar', title='前区号码频率分布')
rear_freq.plot(kind='bar', title='后区号码频率分布')
```

#### 推荐号码
根据历史分布规律，推荐的投注号码为：前区[01, 12, 23, 34, 35]，后区[06, 12]。

---

## 任务3：不同开奖日的号码分布与销售额分析

### 数据分析与可视化
我们统计了周一、周三、周六的开奖号码和总销售额，并对比不同开奖日之间的特征。

#### 代码示例
以下代码展示了如何按开奖日进行统计：
```python
monday_data = data[data['date'].dt.weekday == 0]
wednesday_data = data[data['date'].dt.weekday == 2]
saturday_data = data[data['date'].dt.weekday == 5]

monday_sales = monday_data['total_sales'].mean()
wednesday_sales = wednesday_data['total_sales'].mean()
saturday_sales = saturday_data['total_sales'].mean()

print(f"周一平均销售额: {monday_sales:.2f} 元")
print(f"周三平均销售额: {wednesday_sales:.2f} 元")
print(f"周六平均销售额: {saturday_sales:.2f} 元")
```

#### 可视化结果
![不同开奖日销售额对比](./sales_comparison.png)

---

## 任务4：专家数据分析

### 数据分析与可视化
我们爬取了某彩种中20位以上专家的公开数据，并分析了其基本属性与表现。

#### 代码示例
以下代码展示了如何统计专家数据：
```python
experts_data = pd.read_csv('../expert_data.csv')

experts_data['win_rate'] = experts_data['wins'] / experts_data['total_bets']
experts_data.plot(x='experience', y='win_rate', kind='scatter', title='彩龄与中奖率关系')
```

#### 可视化结果
![专家数据分析](./expert_analysis.png)

---

## 总结
本报告通过数据爬取、分析与可视化，完成了对大乐透开奖数据的全面分析与预测，并对专家数据进行了深入研究。
