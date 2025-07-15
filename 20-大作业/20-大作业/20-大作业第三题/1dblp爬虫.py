import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm


def scrape_top_conferences():
    """爬取KDD/ICML/AAAI正式论文（2020-2025）"""
    conferences = {
        'kdd': range(2020, 2025),
        'icml': range(2020, 2025),
        'aaai': range(2021, 2026)  # AAAI 2020年页面结构不同，从2021开始
    }

    skip_keywords = [
        'Tutorial', 'Workshop', 'Demo', 'New Faculty Highlights',
        'Health Day Papers', 'Abstract', 'Keynote',
        'Panel', 'Summary', 'Invited',' SIGAI Doctoral Consortium',
        'Refine list', 'SPARQL queries'
    ]

    all_papers = []

    for conf, years in conferences.items():
        for year in tqdm(years, desc=f"Processing {conf.upper()}"):
            try:
                url = f"https://dblp.org/db/conf/{conf}/{conf}{year}.html"
                headers = {"User-Agent": "Mozilla/5.0"}

                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')

                # 新版解析逻辑 - 更稳健的section提取
                for header in soup.find_all(['h1', 'h2', 'h3', 'header']):
                    section_title = header.get_text(strip=True)

                    # 跳过非论文section
                    if any(kw in section_title for kw in skip_keywords):
                        continue

                    # 查找关联论文
                    next_elem = header.next_sibling
                    while next_elem:
                        if next_elem.name == 'ul' and 'publ-list' in next_elem.get('class', []):
                            for li in next_elem.find_all('li', class_='entry inproceedings'):
                                title = li.find('span', class_='title')
                                if title:
                                    authors = [a.get_text(strip=True)
                                               for a in li.find_all('span', itemprop='author')]
                                    link = li.find('a', itemprop='url')

                                    all_papers.append({
                                        'conference': conf.upper(),
                                        'year': year,
                                        'section': section_title,
                                        'title': title.get_text(strip=True),
                                        'authors': ', '.join(authors),
                                        'link': link['href'] if link else None
                                    })
                        elif next_elem.name in ['h1', 'h2', 'h3', 'header']:
                            break

                        next_elem = next_elem.next_sibling

                time.sleep(5)  # 增加延迟防止封禁

            except Exception as e:
                print(f"Error processing {conf.upper()} {year}: {str(e)}")
                continue

    # 创建DataFrame并处理空section
    df = pd.DataFrame(all_papers)

    # 如果section列不存在则创建（兼容性处理）
    if 'section' not in df.columns:
        df['section'] = 'Main Conference'

    # 过滤非正式论文
    mask = df['section'].str.contains(
        '|'.join(skip_keywords),
        case=False,
        na=False
    )
    filtered = df[~mask].drop_duplicates(
        subset=['conference', 'year', 'title']
    )

    return filtered.sort_values(['conference', 'year'])


# 执行爬取
print("开始爬取KDD/ICML/AAAI论文（2020-2025）...")
papers_df = scrape_top_conferences()

# 检查结果
if not papers_df.empty:
    print(f"成功爬取 {len(papers_df)} 篇论文")
    print(papers_df.head())
    papers_df.to_csv("conference_papers_2020-2025.csv", index=False, encoding='utf-8-sig')
else:
    print("未能获取数据，请检查HTML结构")



###############仅爬取AAAI2020
# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time
# from tqdm import tqdm
#
#
# def scrape_dblp_conference_with_sections(conf_acronym, years=range(2020, 2025)):
#     base_url = "https://dblp.org/db/conf/{}/{}{}.html"
#     papers = []
#
#     # 需要跳过的分类关键词
#     skip_keywords = ['Tutorial', 'Summary', 'Workshop', 'Demo',  'Abstract', 'Keynote']#'Poster',
#
#     for year in years:
#         print(f"\n正在处理 {conf_acronym.upper()} {year}")
#         url = base_url.format(conf_acronym, conf_acronym, year)
#         try:
#             response = requests.get(url, headers={
#                 "User-Agent": "Mozilla/5.0",
#                 "Accept-Language": "en-US,en;q=0.9"
#             })
#             response.raise_for_status()
#
#             soup = BeautifulSoup(response.text, 'html.parser')
#
#             # 查找所有可能的分类标题
#             sections = soup.find_all(['h2', 'header'], class_=lambda x: x != 'refine-list')
#
#             if not sections:
#                 print(f"警告：未找到分类标题，尝试备用方案")
#                 sections = soup.find_all(['h1', 'h2', 'h3'])
#
#             for section in sections:
#                 section_title = section.get_text(strip=True)
#
#                 # 跳过不需要的分类
#                 if any(keyword in section_title for keyword in skip_keywords):
#                     print(f"跳过分类: {section_title}")
#                     continue
#
#                 print(f"\n处理分类: {section_title}")
#
#                 # 查找关联论文
#                 next_node = section.next_sibling
#                 while next_node:
#                     if next_node.name == 'ul' and 'publ-list' in next_node.get('class', []):
#                         for paper in next_node.find_all('li', class_='entry inproceedings'):
#                             try:
#                                 title = paper.find('span', class_='title').get_text(strip=True)
#                                 authors = [a.get_text(strip=True)
#                                            for a in paper.find_all('span', itemprop='author')]
#                                 link = paper.find('a', itemprop='url')['href'] if paper.find('a',
#                                                                                              itemprop='url') else None
#
#                                 papers.append({
#                                     'conference': conf_acronym.upper(),
#                                     'year': year,
#                                     'section': section_title,
#                                     'title': title,
#                                     'authors': authors,
#                                     'link': link
#                                 })
#                                 print(f"  ✓ {title[:50]}...")
#                             except Exception as e:
#                                 print(f"  论文解析错误: {e}")
#                     elif next_node.name in ['h1', 'h2', 'h3', 'header']:
#                         break
#
#                     next_node = next_node.next_sibling
#
#             time.sleep(5)
#
#         except Exception as e:
#             print(f"处理 {conf_acronym} {year} 时出错: {str(e)}")
#             continue
#
#     return pd.DataFrame(papers)
#
# # 测试单个会议
# test_conf = 'aaai'
# test_years = [2020]  # 先测试最近一年
#
# # 爬取KDD会议数据（自动跳过Tutorials等分类）
# df_kdd = scrape_dblp_conference_with_sections('aaai', [2020])
#
# if not df_kdd.empty:
#     print(f"\n成功获取 {len(df_kdd)} 篇论文:")
#     print(df_kdd[['section', 'title']].head())
#     df_kdd.to_csv("aaai_2020_1.csv", index=False, encoding='utf-8-sig')
# else:
#     print("未能获取数据，请检查")