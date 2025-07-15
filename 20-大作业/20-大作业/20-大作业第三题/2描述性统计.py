#
import pandas as pd

# 读取CSV文件
df = pd.read_csv('all_filter.csv')

# 1. 每年各会议论文总数
yearly_conference_counts = df.groupby(['year', 'conference']).size().unstack(fill_value=0)

# 2. 每年各会议各分类的论文总数
yearly_conference_section_counts = df.groupby(['year', 'conference', 'section']).size().unstack(fill_value=0)

# 打印结果
print("=== 每年各会议论文总数 ===")
print(yearly_conference_counts)

print("\n=== 每年各会议各分类的论文总数 ===")
print(yearly_conference_section_counts)

# 保存结果到Excel文件（更易读）
# with pd.ExcelWriter('conference_stats.xlsx') as writer:
#     yearly_conference_counts.to_excel(writer, sheet_name='会议年度统计')
#     yearly_conference_section_counts.to_excel(writer, sheet_name='会议分类年度统计')

#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error
# from statsmodels.tsa.arima.model import ARIMA
# from prophet import Prophet
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
#
# # 读取数据
# df = pd.read_csv('all_filter.csv')
#
# # 按会议和年份统计论文数量
# conf_year_counts = df.groupby(['conference', 'year']).size().unstack(fill_value=0)
#
# # 选择有足够数据的会议（至少3年数据）
# valid_confs = [conf for conf in conf_year_counts.index if conf_year_counts.loc[conf].count() >= 3]
# if not valid_confs:
#     raise ValueError("没有足够数据（至少需要3年数据）进行模型比较")
#
# # 选择第一个符合条件的会议进行演示
# target_conf = valid_confs[0]
# print(f"\n=== 正在分析会议: {target_conf} ===")
#
# # 准备数据
# years = conf_year_counts.columns.astype(int)
# counts = conf_year_counts.loc[target_conf].values
# X = years.values.reshape(-1, 1)
# y = counts
#
# # 划分训练/测试集（最后一年作为测试）
# X_train, X_test = X[:-1], X[-1:]
# y_train, y_test = y[:-1], y[-1:]
#
# # --- 模型1：线性回归 ---
# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)
# lr_pred = lr_model.predict(X_test)
# lr_mae = mean_absolute_error(y_test, lr_pred)
#
# # --- 模型2：多项式回归（二次）---
# poly_model = make_pipeline(
#     PolynomialFeatures(degree=2),
#     LinearRegression()
# )
# poly_model.fit(X_train, y_train)
# poly_pred = poly_model.predict(X_test)
# poly_mae = mean_absolute_error(y_test, poly_pred)
#
# # --- 模型3：ARIMA ---
# try:
#     arima_model = ARIMA(y_train, order=(1,1,1))  # (p,d,q)
#     arima_fit = arima_model.fit()
#     arima_pred = arima_fit.forecast(steps=1)
#     arima_mae = mean_absolute_error(y_test, arima_pred)
# except Exception as e:
#     print(f"ARIMA失败: {str(e)}")
#     arima_mae = np.inf
#
# # --- 模型4：Prophet ---
# prophet_df = pd.DataFrame({
#     'ds': pd.to_datetime(years[:-1], format='%Y'),
#     'y': y_train
# })
# prophet_model = Prophet(yearly_seasonality=True)
# prophet_model.fit(prophet_df)
# future = prophet_model.make_future_dataframe(periods=1, freq='Y')
# prophet_pred = prophet_model.predict(future)['yhat'].iloc[-1]
# prophet_mae = mean_absolute_error(y_test, [prophet_pred])
#
# # 结果比较
# results = pd.DataFrame({
#     'Model': ['Linear Regression', 'Polynomial (deg=2)', 'ARIMA(1,1,1)', 'Prophet'],
#     'MAE': [lr_mae, poly_mae, arima_mae, prophet_mae],
#     'Prediction': [lr_pred[0], poly_pred[0], arima_pred[0], prophet_pred]
# }, index=np.arange(4) + 1)
#
# print("\n=== 模型性能比较 ===")
# print(results.sort_values('MAE'))
#
# # 可视化
# plt.figure(figsize=(12, 6))
# plt.plot(X_train, y_train, 'ko-', label='Training Data')
# plt.plot(X_test, y_test, 'ro', markersize=10, label='Actual Test Value')
#
# # 绘制预测值
# models = {
#     'Linear': (X_test, lr_pred, 'blue'),
#     'Polynomial': (X_test, poly_pred, 'green'),
#     'ARIMA': (X_test, arima_pred, 'purple'),
#     'Prophet': (X_test, [prophet_pred], 'orange')
# }
#
# for name, (x, pred, color) in models.items():
#     plt.plot(x, pred, 's', color=color, markersize=8, label=f'{name} Prediction')
#
# plt.title(f'Conference "{target_conf}" Paper Count Prediction\n(Test Year: {X_test[0][0]}, Actual: {y_test[0]})')
# plt.xlabel('Year')
# plt.ylabel('Paper Count')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

####result:
# === 模型性能比较 ===
#                 Model         MAE   Prediction
# 3        ARIMA(1,1,1)   82.826390  3011.173610
# 4             Prophet  380.731221  2713.268779
# 2  Polynomial (deg=2)  456.999999  3550.999999
# 1   Linear Regression  639.500000  2454.500000