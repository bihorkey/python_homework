#################################################6预测第四题

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_data(filepath='all_filter.csv'):
    """加载数据并验证格式"""
    df = pd.read_csv(filepath)

    # 列名标准化
    required_cols = {'title', 'conference', 'year'}
    col_map = {col.lower(): col for col in df.columns}

    missing = required_cols - set(col_map.keys())
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    return df.rename(columns={
        col_map['title']: 'title',
        col_map['conference']: 'conference',
        col_map['year']: 'year'
    })


def predict_conference_papers(df, conference, predict_year, min_years=3):
    """预测会议论文数量（带可视化）"""
    # 准备数据
    conf_df = df[df['conference'] == conference]
    year_counts = conf_df['year'].value_counts().sort_index()

    if len(year_counts) < min_years:
        print(f"{conference}数据不足（需至少{min_years}年数据）")
        return None

    years = year_counts.index.values
    counts = year_counts.values

    # 线性回归预测
    lin_reg = LinearRegression()
    lin_reg.fit(years.reshape(-1, 1), counts)
    linear_pred = int(lin_reg.predict([[predict_year]])[0])

    # ARIMA预测
    try:
        arima_model = ARIMA(counts, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        arima_pred = int(arima_fit.forecast(steps=1)[0])
    except Exception as e:
        print(f"ARIMA模型失败: {str(e)}")
        arima_pred = linear_pred

    final_pred = int(np.mean([linear_pred, arima_pred]))

    # 增强可视化
    plt.figure(figsize=(10, 6), dpi=120)
    ax = plt.gca()

    # 绘制历史数据
    line, = plt.plot(years, counts, 'bo-', linewidth=2, markersize=8,
                     label='Historical data')

    # 标注历史数值
    for x, y in zip(years, counts):
        plt.text(x, y + 20, f'{y}', ha='center', va='bottom', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 绘制预测数据
    pred_x = np.append(years, predict_year)
    pred_y = np.append(counts, final_pred)
    plt.plot(pred_x[-2:], pred_y[-2:], 'r--', linewidth=2, label='The predicted trend')
    plt.scatter(predict_year, final_pred, c='red', s=100, zorder=5)

    # 标注预测值
    plt.text(predict_year, final_pred + 20, f'predict: {final_pred}',
             ha='center', va='bottom', fontsize=10, color='red',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 图表美化
    ax.set_title(f"{conference}Prediction of the number of accepted papers ({predict_year})", pad=20)
    ax.set_xlabel("year", labelpad=10)
    ax.set_ylabel("number", labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', framealpha=0.9)

    # 设置x轴为整数年份
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 自动调整y轴范围
    y_min = min(counts) * 0.9
    y_max = max(max(counts), final_pred) * 1.1
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{conference}_paper_count_prediction.png", bbox_inches='tight')
    plt.close()

    return final_pred


def main():
    try:
        # 加载数据
        print("正在加载数据...")
        df = load_data('all_filter.csv')

        # 检查数据年份范围
        print("\n数据年份分布:")
        print(df.groupby('conference')['year'].agg(['min', 'max', 'count']))

        # 预测配置
        prediction_config = [
            ('AAAI', 2026),
            ('KDD', 2025),
            ('ICML', 2025)
        ]

        # 执行预测
        print("\n=== 论文收录量预测 ===")
        results = []
        for conf, year in prediction_config:
            pred = predict_conference_papers(df, conf, year)
            if pred:
                results.append((conf, year, pred))
                print(f"{conf} {year}年预测: {pred}篇")
                print(f"图表已保存为: {conf}_paper_count_prediction.png")

        # 生成汇总表格
        if results:
            result_df = pd.DataFrame(results, columns=['会议', '年份', '预测论文量'])
            print("\n预测结果汇总:")
            print(result_df.to_string(index=False))

    except Exception as e:
        print(f"运行错误: {str(e)}")


if __name__ == "__main__":
    main()
####result
# 预测结果汇总:
#   会议   年份  预测论文量
# AAAI 2026   3140
#  KDD 2025    614
# ICML 2025   2880