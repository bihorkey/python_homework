import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 加载数据
df = pd.read_csv('all_filter.csv')

# 按会议和年份统计论文数量
paper_count = df.groupby(['conference', 'year']).size().reset_index(name='count')

# 绘制趋势图
plt.figure(figsize=(12, 6))
sns.lineplot(data=paper_count, x='year', y='count', hue='conference',
             marker='o', linewidth=2.5)

plt.title('Paper Publication Trends in Top CS Conferences (2020-2025)')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.legend(title='Conference', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('publication_trends.png', dpi=300, bbox_inches='tight')
plt.show()