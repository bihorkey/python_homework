import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib

# 强制使用Agg后端（无图形界面），避免Windows崩溃问题
matplotlib.use('Agg')

def setup_fonts():
    """设置中文字体支持，确保中文正常显示"""
    try:
        # Windows系统常见中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "FangSong"]
        # Mac系统常见中文字体
        # plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Heiti SC"]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.rcParams['figure.dpi'] = 150  # 降低绘图DPI减少内存占用
        plt.rcParams['savefig.dpi'] = 300  # 保存时使用高DPI
    except Exception as e:
        print(f"字体设置警告: {str(e)}")
        # 如果设置失败，尝试使用系统默认字体
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # 适用于Mac

def load_and_preprocess_data(file_path):
    """加载并预处理数据，解决警告和数据格式问题"""
    try:
        # 读取数据
        df = pd.read_csv(file_path)
        print(f"数据加载成功，共{len(df)}条记录")

        # 年龄处理
        df['年龄'] = pd.to_numeric(df['年龄'], errors='coerce')
        df = df.dropna(subset=['年龄'])
        print(f"\n年龄数据统计:\n{df['年龄'].describe()}")

        # 性别处理
        gender_map = {'先生': '男', '女士': '女'}
        df['性别'] = df['性别'].map(gender_map)
        df = df[df['性别'].isin(['男', '女'])]  # 只保留有效性别
        print(f"\n性别分布:\n{df['性别'].value_counts()}")

        # 出生地处理（提取省份）
        df['省份'] = None
        # 处理"中国-XX-XX"格式
        pattern1 = df['出生地_中文'].str.match(r'^中国-[^-]+-[^-]+$', na=False)
        df.loc[pattern1, '省份'] = df.loc[pattern1, '出生地_中文'].str.split('-').str[1]
        # 处理"中国-XX"格式
        pattern2 = df['出生地_中文'].str.match(r'^中国-[^-]+$', na=False)
        df.loc[pattern2, '省份'] = df.loc[pattern2, '出生地_中文'].str.split('-').str[1]
        # 处理其他格式（取前2个字符）
        other_pattern = ~pattern1 & ~pattern2 & df['出生地_中文'].notna()
        df.loc[other_pattern, '省份'] = df.loc[other_pattern, '出生地_中文'].str[:2]

        print(f"\n省份分布（前5）:\n{df['省份'].value_counts().head(5)}")
        return df
    except Exception as e:
        print(f"数据处理错误: {str(e)}")
        return None

def create_age_distribution_chart(df, output_dir):
    """生成年龄分布直方图"""
    try:
        plt.figure(figsize=(10, 6))
        # 限制年龄范围避免异常值
        age_data = df[(df['年龄'] >= 30) & (df['年龄'] <= 90)]['年龄']
        sns.histplot(age_data, bins=12, color='skyblue', kde=True)
        plt.title('富豪年龄分布')
        plt.xlabel('年龄')
        plt.ylabel('人数')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
        plt.close()
        print("年龄分布图表生成成功")
    except Exception as e:
        print(f"年龄图表生成失败: {str(e)}")

def create_gender_distribution_chart(df, output_dir):
    """生成性别分布饼图"""
    try:
        plt.figure(figsize=(8, 8))
        gender_counts = df['性别'].value_counts()
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
                startangle=90, colors=['#66b3ff', '#ff9999'],
                wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        plt.title('富豪性别分布')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
        plt.close()
        print("性别分布图表生成成功")
    except Exception as e:
        print(f"性别图表生成失败: {str(e)}")

def create_birthplace_distribution_chart(df, output_dir):
    """生成出生地分布柱状图"""
    try:
        plt.figure(figsize=(12, 6))
        province_counts = df['省份'].value_counts().head(10)
        # 修改后的绘图代码
        sns.barplot(x=province_counts.index,
                   y=province_counts.values,
                   hue=province_counts.index,
                   palette='viridis',
                   legend=False)
        plt.title('富豪出生地省份TOP10')
        plt.xlabel('省份')
        plt.ylabel('人数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'birthplace_distribution.png'))
        plt.close()
        print("出生地分布图表生成成功")
    except Exception as e:
        print(f"出生地图表生成失败: {str(e)}")

def create_age_wealth_relationship_chart(df, output_dir):
    """生成年龄与财富关系散点图"""
    try:
        plt.figure(figsize=(10, 6))
        # 限制财富值范围避免异常值影响
        wealth_data = df[df['财富值_人民币_亿'] < 5000]
        sns.scatterplot(x='年龄', y='财富值_人民币_亿', data=wealth_data,
                        alpha=0.6, color='purple', s=50)
        plt.title('年龄与财富关系')
        plt.xlabel('年龄')
        plt.ylabel('财富值(亿人民币)')
        plt.grid(axis='both', alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_wealth_relationship.png'))
        plt.close()
        print("年龄与财富关系图表生成成功")
    except Exception as e:
        print(f"年龄与财富图表生成失败: {str(e)}")

def main():
    """主函数：协调数据处理和图表生成"""
    DATA_FILE = '2024胡润百富榜.csv'  # 请替换为实际文件路径
    OUTPUT_DIR = 'charts'

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 设置字体
    setup_fonts()

    # 加载和预处理数据
    df = load_and_preprocess_data(DATA_FILE)
    if df is None:
        print("数据加载失败，程序退出")
        return

    # 生成图表
    create_age_distribution_chart(df, OUTPUT_DIR)
    create_gender_distribution_chart(df, OUTPUT_DIR)
    create_birthplace_distribution_chart(df, OUTPUT_DIR)
    create_age_wealth_relationship_chart(df, OUTPUT_DIR)

    print(f"\n所有图表已成功生成并保存至 '{OUTPUT_DIR}' 目录")

if __name__ == "__main__":
    main()