############################################################################3预测图不对
# 第三题只用其预测热点
import pandas as pd
import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# 初始化NLTK
nltk.download('stopwords')
stemmer = PorterStemmer()

# 自定义停用词
stop_words = set(stopwords.words('english'))
custom_stopwords = {'using', 'based', 'via', 'approach', 'method',
                    'model', 'system', 'learning', 'network', 'analysis',
                    'algorithm', 'framework', 'task', 'problem'}
stop_words.update(custom_stopwords)


def load_and_preprocess(filepath='all_filter.csv'):
    """直接加载并预处理CSV文件"""
    # 自动检测列名（兼容不同大小写）
    df = pd.read_csv(filepath)

    # 列名标准化
    col_map = {col.lower(): col for col in df.columns}
    required_columns = {'title', 'conference', 'year'}

    # 验证必要列是否存在
    if not required_columns.issubset(col_map.keys()):
        missing = required_columns - set(col_map.keys())
        raise ValueError(f"CSV文件缺少必要列: {missing}")

    # 标准化列名
    df = df.rename(columns={
        col_map['title']: 'title',
        col_map['conference']: 'conference',
        col_map['year']: 'year'
    })

    # 预处理标题
    def clean_title(title):
        if not isinstance(title, str):
            return ""
        title = re.sub(r'[^\w\s-]', '', title.lower())  # 保留连字符
        words = []
        for w in title.split():
            w = w.strip('-')
            if (len(w) > 2 and w not in stop_words
                    and not w.isnumeric() and w.replace('-', '').isalpha()):
                words.append(w)
        return ' '.join(words)

    df['processed_title'] = df['title'].apply(clean_title)
    return df


def extract_keywords(titles, ngram_range=(1, 3), top_n=50):
    """改进的关键词提取"""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=2000,
        stop_words=None,
        min_df=3
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(titles)
        terms = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

        # 组合统计量
        count_vec = CountVectorizer(ngram_range=ngram_range, min_df=3)
        count_matrix = count_vec.fit_transform(titles)
        counts = np.asarray(count_matrix.sum(axis=0)).ravel()

        df = pd.DataFrame({
            'term': terms,
            'tfidf': scores,
            'count': counts,
            'score': scores * np.log1p(counts)  # 平衡权重
        }).sort_values('score', ascending=False)

        # 过滤通用术语
        generic_terms = {'deep learning', 'machine learning', 'neural network'}
        return [(row['term'], row['score'])
                for _, row in df[~df['term'].isin(generic_terms)].head(top_n).iterrows()]
    except Exception as e:
        print(f"关键词提取错误: {str(e)}")
        return []


def analyze_conference(df, conference, output_dir="output"):
    """完整会议分析流程"""
    os.makedirs(output_dir, exist_ok=True)
    conf_df = df[df['conference'] == conference].copy()

    # 按年份处理
    yearly_keywords = {}
    for year, group in conf_df.groupby('year'):
        titles = group['processed_title'].tolist()
        yearly_keywords[year] = extract_keywords(titles)

    # 时间加权预测
    def get_weighted_keywords(yearly_data, decay=0.6):
        weighted = defaultdict(float)
        years = sorted(yearly_data.keys(), reverse=True)
        weights = [decay ** i for i in range(len(years))]
        weights = [w / sum(weights) for w in weights]  # 归一化

        for year, weight in zip(years, weights):
            for term, score in yearly_data[year]:
                weighted[term] += score * weight

        return sorted(weighted.items(), key=lambda x: -x[1])

    # 生成预测
    weighted_terms = get_weighted_keywords(yearly_keywords)

    # 过滤常见词（出现3年以上的）
    term_freq = defaultdict(int)
    for year in yearly_keywords:
        for term, _ in yearly_keywords[year]:
            term_freq[term] += 1

    predictions = [
                      term for term, score in weighted_terms
                      if term_freq.get(term, 0) < 3
                  ][:5]

    # 可视化
    plt.figure(figsize=(12, 6))
    for term in predictions[:3]:  # 展示前3个预测项的趋势
        years = []
        scores = []
        for year in sorted(yearly_keywords.keys()):
            score = next((s for t, s in yearly_keywords[year] if t == term), 0)
            years.append(year)
            scores.append(score)
        plt.plot(years, scores, 'o-', label=term)

    plt.title(f"{conference} Emerging Trends Prediction")
    plt.xlabel("Year")
    plt.ylabel("Trend Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/{conference}_trends.png", dpi=120, bbox_inches='tight')
    plt.close()

    # 生成词云
    if weighted_terms:
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            collocations=False,
            max_words=100
        ).generate_from_frequencies(dict(weighted_terms[:50]))

        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(f"{output_dir}/{conference}_wordcloud.png", dpi=150, bbox_inches='tight')
        plt.close()

    return predictions


def main():
    # 自动加载并预处理数据
    try:
        print("Loading data from all_filter.csv...")
        df = load_and_preprocess('all_filter.csv')

        # 检查数据
        print("\n数据概览:")
        print(f"总论文数: {len(df)}")
        print("会议分布:")
        print(df['conference'].value_counts())
        print("年份分布:")
        print(df['year'].value_counts().sort_index())

        # 分析各会议
        conferences = sorted(df['conference'].unique())
        predictions = {}

        for conf in conferences:
            print(f"\nAnalyzing {conf}...")
            pred = analyze_conference(df, conf)
            predictions[conf] = pred

        # 打印预测结果
        print("\n=== 最终预测结果 ===")
        for conf, preds in predictions.items():
            print(f"\n{conf} 2026:" if "AAAI" in conf else f"\n{conf} 2025:")
            for i, term in enumerate(preds, 1):
                print(f"{i}. {term}")

        print("\n分析完成，结果已保存到/output目录")

    except Exception as e:
        print(f"运行错误: {str(e)}")
        print("请确保: 1. all_filter.csv存在 2. 包含title/conference/year列")


if __name__ == "__main__":
    main()
# #####result
# D:\学习\软件工程\python与数分\test\.venv\Scripts\python.exe D:\学习\软件工程\python与数分\test\练习.py
# [nltk_data] Downloading package stopwords to
# [nltk_data]     C:\Users\Qiujuan\AppData\Roaming\nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
# Loading data from all_filter.csv...
#
# 数据概览:
# 总论文数: 22682
# 会议分布:
# conference
# AAAI    12503
# ICML     7940
# KDD      2239
# Name: count, dtype: int64
# 年份分布:
# year
# 2020    3127
# 2021    3366
# 2022    3143
# 2023    4130
# 2024    5822
# 2025    3094
# Name: count, dtype: int64
#
# Analyzing AAAI...
# 关键词提取错误: operands could not be broadcast together with shapes (2000,) (2313,)
#
# Analyzing ICML...
#
# Analyzing KDD...
#
# === 最终预测结果 ===
#
# AAAI 2026:
# 1. diffusion
# 2. federated
# 3. label
# 4. guided
# 5. large
#
# ICML 2025:
# 1. language models
# 2. large
# 3. diffusion
# 4. large language
# 5. generation
#
# KDD 2025:
# 1. language
# 2. time series
# 3. robust
# 4. large language
# 5. cross

# 分析完成，结果已保存到/output目录