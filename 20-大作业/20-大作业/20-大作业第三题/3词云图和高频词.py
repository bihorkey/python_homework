from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
import os

from dblp论文数量趋势 import df

# 初始化NLTK停用词
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
custom_stopwords = {'using', 'based', 'via', 'approach', 'method', 'model', 'system', 'learning', 'network'}
stop_words.update(custom_stopwords)


# 预处理标题（保留连字符和词组）
def preprocess_title(title):
    if not isinstance(title, str):
        return ""

    # 保留字母、数字和连字符
    title = re.sub(r'[^\w\s-]', '', title.lower())
    # 分词并过滤
    words = [w for w in title.split()
             if (w.replace('-', '').isalpha())
             and (w not in stop_words)
             and (len(w) > 2)]
    return ' '.join(words)


# 改进的关键词提取函数
def extract_keywords_by_group(df, group_cols, top_n=30):
    # 使用更大的n-gram范围(1-3)和更高的min_df
    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=2000,
                                 stop_words=None, min_df=3)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=2000,
                                       stop_words=None, min_df=3)

    groups = df.groupby(group_cols)
    keywords_dict = defaultdict(dict)

    for name, group in groups:
        if len(group) < 5:  # 忽略数据量太小的分组
            continue

        titles = group['processed_title'].tolist()
        titles = [t for t in titles if t.strip()]  # 移除空字符串

        if not titles:
            continue

        try:
            # 1. 统计词频
            count_matrix = vectorizer.fit_transform(titles)
            count_terms = vectorizer.get_feature_names_out()
            count_scores = np.asarray(count_matrix.sum(axis=0)).ravel()

            # 2. 统计TF-IDF权重
            tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
            tfidf_terms = tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

            # 3. 合并统计结果
            stats_df = pd.DataFrame({
                'term': count_terms,
                'count': count_scores,
                'tfidf': tfidf_scores,
                'combined_score': count_scores * tfidf_scores
            }).sort_values('combined_score', ascending=False)

            # 4. 过滤通用术语
            generic_terms = {'deep learning', 'machine learning', 'neural network'}
            stats_df = stats_df[~stats_df['term'].isin(generic_terms)]

            # 5. 保存最终关键词
            top_keywords = [(row['term'], row['combined_score'])
                            for _, row in stats_df.head(top_n).iterrows()]
            keywords_dict[name] = top_keywords

            # 6. 输出分析结果
            print(f"\n=== {name} ===")
            print("代表性标题:", titles[:3])
            print("Top 10关键词:")
            for term, score in top_keywords[:10]:
                print(f"{term}: {score:.2f}")

        except Exception as e:
            print(f"处理{name}时出错:", str(e))

    return keywords_dict


# 改进的词云生成
def generate_wordcloud(keywords_dict, conference, year, output_dir="wordclouds"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    keywords = keywords_dict.get((conference, year), [])
    if not keywords:
        print(f"没有找到{conference} {year}的数据")
        return

    word_freq = {word: score for word, score in keywords}

    wordcloud = WordCloud(
        width=1400,
        height=700,
        background_color='white',
        colormap='plasma',
        collocations=True,
        prefer_horizontal=0.9,
        max_words=100,
        min_font_size=10,
        max_font_size=200
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Research Hotspots in {conference} {year}', fontsize=18, pad=20)
    plt.axis('off')
    plt.tight_layout()

    # 保存词云图
    filename = f"{output_dir}/{conference}_{year}_hotspots.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，避免内存泄漏
    print(f"词云图已保存到: {filename}")


# 预测热点函数
def predict_hotspots(keywords_dict, conference, current_year, predict_year):
    # 获取最近两年的数据
    current_data = keywords_dict.get((conference, current_year), [])
    previous_data = keywords_dict.get((conference, current_year - 1), [])

    if not current_data:
        print(f"没有找到{conference} {current_year}的数据")
        return

    # 提取当前年和上一年的关键词
    current_terms = {term for term, _ in current_data[:20]}
    previous_terms = {term for term, _ in previous_data[:20]} if previous_data else set()

    # 找出新出现的关键词
    emerging_terms = current_terms - previous_terms

    # 从当前数据中筛选新兴术语及其分数
    emerging_hotspots = [(term, score) for term, score in current_data if term in emerging_terms]

    # 按分数排序
    emerging_hotspots.sort(key=lambda x: x[1], reverse=True)

    # 预测前5个热点
    predicted_hotspots = [term for term, score in emerging_hotspots[:5]]

    print(f"\n预测{predict_year}年{conference}可能的研究热点:")
    for i, hotspot in enumerate(predicted_hotspots, 1):
        print(f"{i}. {hotspot}")

    return predicted_hotspots

##效果不好
def merge_similar_keywords(keywords_list, similarity_threshold=0.8):
    from difflib import SequenceMatcher

    merged = []
    skip_indices = set()

    for i, (term1, score1) in enumerate(keywords_list):
        if i in skip_indices:
            continue

        # 查找相似术语
        similar_terms = [(term1, score1)]
        for j, (term2, score2) in enumerate(keywords_list[i + 1:], start=i + 1):
            ratio = SequenceMatcher(None, term1, term2).ratio()
            if ratio > similarity_threshold:
                similar_terms.append((term2, score2))
                skip_indices.add(j)

        # 合并相似术语（保留最长版本）
        merged_term = max(similar_terms, key=lambda x: len(x[0]))[0]
        merged_score = sum(score for _, score in similar_terms)
        merged.append((merged_term, merged_score))

    return sorted(merged, key=lambda x: x[1], reverse=True)

# 后加，效果不好
def improved_predict_hotspots(keywords_dict, conference, current_year, predict_year):
    # 获取数据并合并相似关键词
    current_data = merge_similar_keywords(
        keywords_dict.get((conference, current_year), []))
    previous_data = merge_similar_keywords(
        keywords_dict.get((conference, current_year - 1), [])) if current_year > 2020 else []

    # 提取术语集合（考虑词干化）
    stemmer = nltk.stem.PorterStemmer()

    def get_stemmed_set(terms):
        return {stemmer.stem(term.split()[-1]) for term in terms}  # 只取最后一个词干

    current_terms = {term for term, _ in current_data[:30]}
    previous_terms = {term for term, _ in previous_data[:30]} if previous_data else set()

    # 更智能的新兴术语检测
    emerging_terms = []
    for term, score in current_data:
        stemmed_parts = [stemmer.stem(w) for w in term.split()]
        # 排除纯形容词组合
        if len(stemmed_parts) == 1 and stemmed_parts[0] in ['large', 'high']:
            continue
        # 检查是否真正新兴
        is_emerging = all(
            stemmer.stem(w) not in get_stemmed_set(previous_terms)
            for w in term.split()[-2:]  # 检查最后两个词
        )
        if is_emerging:
            emerging_terms.append((term, score))

    # 按分数排序并选取独特概念
    predicted = []
    seen_stems = set()
    for term, score in sorted(emerging_terms, key=lambda x: x[1], reverse=True):
        term_stem = ' '.join(stemmer.stem(w) for w in term.split())
        if term_stem not in seen_stems and len(predicted) < 5:
            predicted.append(term)
            seen_stems.add(term_stem)

    return predicted

# 主函数
def main():
    # 假设df已加载
    if 'df' not in globals():
        print("请先加载数据到df变量中")
        return

    # 预处理标题
    df['processed_title'] = df['title'].apply(preprocess_title)

    # 提取关键词（按会议和年份）
    keywords_by_conf_year = extract_keywords_by_group(df, ['conference', 'year'])

    # 为每个会议生成词云图
    conferences = df['conference'].unique()
    years = sorted(df['year'].unique())

    for conference in conferences:
        for year in years:
            generate_wordcloud(keywords_by_conf_year, conference, year)
    #
    # # 热点预测
    # # 预测2026年AAAI（假设最新数据是2024年）
    # predict_hotspots(keywords_by_conf_year, 'AAAI', 2025, 2026)
    #
    # # 预测2025年KDD和ICML（假设最新数据是2023年）
    # predict_hotspots(keywords_by_conf_year, 'KDD', 2024, 2025)
    # predict_hotspots(keywords_by_conf_year, 'ICML', 2024, 2025)

#后加两函数主函数的变化
    conferences_years = [
        ('AAAI', 2024, 2026),
        ('KDD', 2023, 2025),
        ('ICML', 2023, 2025)
    ]

    # for conf, curr, pred in conferences_years:
    #     print(f"\n=== 改进版预测 {pred}年{conf} 热点 ===")
    #     hotspots = improved_predict_hotspots(keywords_by_conf_year, conf, curr, pred)
    #     for i, term in enumerate(hotspots, 1):
    #         print(f"{i}. {term}")


if __name__ == "__main__":
    main()