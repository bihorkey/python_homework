import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据准备函数
def prepare_data():
    # 读取训练数据（2022-2024年）
    train_data = pd.read_csv('homework/second/data/dalian_weather_2022-2024.csv', encoding='gbk')
    train_data['日期'] = pd.to_datetime(train_data['日期'], format='%Y年%m月%d日')
    train_data['年份'] = train_data['日期'].dt.year
    train_data['月份'] = train_data['日期'].dt.month
    
    # 读取预测数据（2025年）
    predict_data = pd.read_csv('homework/second/data/dalian_weather_2025.csv', encoding='gbk')
    predict_data['日期'] = pd.to_datetime(predict_data['日期'], format='%Y年%m月%d日')
    predict_data['年份'] = predict_data['日期'].dt.year
    predict_data['月份'] = predict_data['日期'].dt.month
    
    return train_data, predict_data

# 2. 特征工程函数
def create_features(df):
    # 基础特征
    df['月份_sin'] = np.sin(2 * np.pi * df['月份'] / 12)
    df['月份_cos'] = np.cos(2 * np.pi * df['月份'] / 12)
    
    # 添加多项式特征（二次项）
    df['月份_sq'] = df['月份'] ** 2
    
    return df

# 3. 主函数
def main():
    # 准备数据
    train_data, predict_data = prepare_data()
    
    # 计算历史月平均温度
    monthly_avg = train_data.groupby(['年份', '月份'])['最高温'].mean().reset_index()
    historical_avg = monthly_avg.groupby('月份')['最高温'].mean().values
    
    # 特征工程
    train_features = create_features(monthly_avg)
    X_train = train_features[['月份_sin', '月份_cos', '月份_sq']]
    y_train = train_features['最高温']
    
    # 准备2025年数据
    future_months = pd.DataFrame({'月份': range(1, 13)})
    future_features = create_features(future_months)
    X_future = future_features[['月份_sin', '月份_cos', '月份_sq']]
    
    # 准备真实数据（2025年1-6月）
    true_2025 = predict_data.groupby(['年份', '月份'])['最高温'].mean().reset_index()
    true_2025 = create_features(true_2025)
    X_test = true_2025[['月份_sin', '月份_cos', '月份_sq']]
    y_test = true_2025['最高温']
    
    # 创建并训练线性回归模型
    # 使用管道组合多项式特征和线性回归
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),  # 添加特征交互项
        LinearRegression()
    )
    
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    print(f"线性回归模型评估:")
    print(f"- 平均绝对误差(MAE): {mae:.2f}°C")
    print(f"- R²分数: {r2:.2f}")
    
    # 查看模型系数（仅限纯线性回归部分）
    if hasattr(model.named_steps['linearregression'], 'coef_'):
        print("\n特征系数:")
        features = ['月份_sin', '月份_cos', '月份_sq']
        if 'polynomialfeatures' in model.named_steps:
            # 获取多项式特征名称
            poly = model.named_steps['polynomialfeatures']
            features = poly.get_feature_names_out(features)
        
        for name, coef in zip(features, model.named_steps['linearregression'].coef_):
            print(f"{name}: {coef:.3f}")
        print(f"截距: {model.named_steps['linearregression'].intercept_:.3f}")
    
    # 预测2025年全年
    y_pred_future = model.predict(X_future)
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    
    # 绘制历史平均值
    plt.plot(range(1, 13), historical_avg, 'g--', label='历史平均值', alpha=0.7)
    
    # 绘制预测值
    plt.plot(range(1, 13), y_pred_future, 'b-o', label='预测温度')
    
    # 绘制真实值
    plt.plot(range(1, 7), y_test, 'r-s', label='真实温度')
    
    # 添加误差区域
    errors = [abs(p - t) for p, t in zip(y_pred_test, y_test)]
    plt.fill_between(range(1, 7), 
                    [p - e for p, e in zip(y_pred_test, errors)],
                    [p + e for p, e in zip(y_pred_test, errors)],
                    color='skyblue', alpha=0.3, label='误差范围')
    
    # 添加标签和标题
    plt.title('2025年温度预测 (线性回归模型)', fontsize=15)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('平均最高温度 (°C)', fontsize=12)
    plt.xticks(range(1, 13))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 添加数据标签
    for i in range(6):
        plt.annotate(f"预测:{y_pred_future[i]:.1f}", 
                    (i+1, y_pred_future[i] + 0.5),
                    ha='center', fontsize=9)
        plt.annotate(f"真实:{y_test[i]:.1f}", 
                    (i+1, y_test[i] - 0.8),
                    ha='center', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig('linear_regression_prediction.png', dpi=300)
    plt.show()
    
    # 保存预测结果
    result_df = pd.DataFrame({
        '月份': range(1, 13),
        '预测温度': y_pred_future
    })
    result_df.to_csv('linear_regression_predictions.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()