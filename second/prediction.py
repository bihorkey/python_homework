import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 数据准备
def prepare_data():
    # 读取训练数据（2022-2024年）
    train_data = pd.read_csv('homework/second/data/dalian_weather_2022-2024.csv', encoding='gbk')
    # 将日期列转换为datetime类型
    train_data['日期'] = pd.to_datetime(train_data['日期'], format='%Y年%m月%d日')
    
    # 提取年份和月份
    train_data['年份'] = train_data['日期'].dt.year
    train_data['月份'] = train_data['日期'].dt.month
    
    # 读取预测数据（2025年）
    predict_data = pd.read_csv('homework/second/data/dalian_weather_2025.csv', encoding='gbk')
    # 将日期列转换为datetime类型
    predict_data['日期'] = pd.to_datetime(predict_data['日期'], format='%Y年%m月%d日')
    
    # 提取年份和月份
    predict_data['年份'] = predict_data['日期'].dt.year
    predict_data['月份'] = predict_data['日期'].dt.month
    
    return train_data, predict_data

# 2. 特征工程 - 处理周期性月份特征
def create_features(df):
    # 添加周期性特征（正弦和余弦变换）
    df['月份_sin'] = np.sin(2 * np.pi * df['月份'] / 12)
    df['月份_cos'] = np.cos(2 * np.pi * df['月份'] / 12)
    
    # 添加月份平方特征（捕捉非线性关系）
    df['月份_sq'] = df['月份'] ** 2
    
    return df

# 3. 训练模型
def train_model(X_train, y_train):
    # 初始化随机森林回归模型
    model = RandomForestRegressor(
        n_estimators=150,  # 树的数量
        max_depth=5,        # 树的最大深度
        random_state=42,    # 随机种子
        min_samples_split=3 # 分裂节点所需最小样本数
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    return model

# 4. 评估模型
def evaluate_model(model, X_test, y_test):
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"模型评估结果:")
    print(f"- 平均绝对误差(MAE): {mae:.2f}°C")
    print(f"- R²分数: {r2:.2f}")
    
    return y_pred

# 5. 可视化结果
def visualize_results(months, true_temps, pred_temps, historical_avg):
    plt.figure(figsize=(12, 7))
    
    # 绘制历史平均值（2022-2024年）
    plt.plot(months, historical_avg, 'g--', linewidth=2, label='历史平均值', alpha=0.7)
    
    # 绘制预测值（2025年1-12月）
    plt.plot(months, pred_temps, 'b-o', linewidth=2, markersize=8, label='预测温度')
    
    # 绘制真实值（2025年1-6月）
    plt.plot(months[:6], true_temps[:6], 'r-s', linewidth=2, markersize=8, label='真实温度')
    
    # 添加标题和标签
    plt.title('2025年月平均最高温度预测 vs 真实值', fontsize=16, pad=20)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('平均最高温度 (°C)', fontsize=12)
    plt.xticks(months)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # 添加数据标签
    for i, month in enumerate(months):
        if i < len(true_temps):
            plt.annotate(f"{true_temps[i]:.1f}", 
                         (month, true_temps[i] - 0.8), 
                         ha='center', fontsize=10, color='red')
        
        plt.annotate(f"{pred_temps[i]:.1f}", 
                     (month, pred_temps[i] + 0.5), 
                     ha='center', fontsize=10, color='blue')
    
    # 添加误差区域（1-6月）
    errors = [abs(p - t) for p, t in zip(pred_temps[:6], true_temps[:6])]
    plt.fill_between(months[:6], 
                     [p - e for p, e in zip(pred_temps[:6], errors)],
                     [p + e for p, e in zip(pred_temps[:6], errors)],
                     color='skyblue', alpha=0.3, label='预测误差范围')
    
    plt.tight_layout()
    plt.savefig('temperature_prediction_comparison.png', dpi=300)
    plt.show()

# 主函数
def main():
    # 1. 准备数据
    train_data, predict_data = prepare_data()
    
    # 2. 计算历史月平均温度（2022-2024年）
    monthly_avg = train_data.groupby(['年份', '月份'])['最高温'].mean().reset_index()
    historical_avg = monthly_avg.groupby('月份')['最高温'].mean().values
    
    # 3. 准备训练数据
    train_features = create_features(monthly_avg)
    X_train = train_features[['月份_sin', '月份_cos', '月份_sq']]
    y_train = train_features['最高温']
    
    # 4. 准备预测数据（2025年1-12月）
    future_months = pd.DataFrame({'月份': range(1, 13)})
    future_features = create_features(future_months)
    X_future = future_features[['月份_sin', '月份_cos', '月份_sq']]
    
    # 5. 准备真实数据（2025年1-6月）
    true_2025 = predict_data.groupby(['年份', '月份'])['最高温'].mean().reset_index()
    true_2025 = create_features(true_2025)
    X_test = true_2025[['月份_sin', '月份_cos', '月份_sq']]
    y_test = true_2025['最高温']
    
    # 6. 训练模型
    model = train_model(X_train, y_train)
    
    # 7. 评估模型（在2025年1-6月数据上）
    y_pred_test = evaluate_model(model, X_test, y_test)
    
    # 8. 预测2025年全年
    y_pred_future = model.predict(X_future)
    
    # 9. 可视化结果
    months = list(range(1, 13))
    visualize_results(months, y_test.values, y_pred_future, historical_avg)
    
    # 10. 保存预测结果
    result_df = pd.DataFrame({
        '月份': months,
        '预测温度': y_pred_future
    })
    result_df.to_csv('2025_temperature_predictions.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()