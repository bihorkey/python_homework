import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import lxml
from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def webparser():
    all_data = []

    opt = webdriver.ChromeOptions()
    opt.add_argument('--blink-settings=imagesEnabled=false')
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=opt)

    driver.get("https://www.zhcw.com/kjxx/dlt/")
    element_recent_100 = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "span.annq[data-z='100']"))
    )
    element_recent_100.click()
    time.sleep(2)

    for i in range(ord('1'), ord('5')):
        sxpath = '//a[@title="' + chr(i) + '"and contains(@avalon-events, "click")]'
        if i > ord('1'):
            element_page = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, sxpath)
                )
            )
            element_page.click()
            time.sleep(2)

        soup = BeautifulSoup(driver.page_source, 'lxml')
        page_data = pageparser(i - ord('0'), soup)
        all_data.extend(page_data)

    driver.quit()

    df = pd.DataFrame(all_data)
    df.to_csv('../lottery_data.csv', index=False, encoding='utf-8-sig')

def pageparser(page_num, soup):
    data = []
    rows = soup.select('table tbody tr')

    for row in rows:
        if not row.select('td'):
            continue

        cells = row.select('td')
        if len(cells) < 14:
            continue

        issue_num = cells[0].text.strip()
        if not issue_num:
            continue

        date_str = cells[1].text.strip()
        front_numbers_cell = cells[2]
        back_numbers_cell = cells[3]

        front_spans = front_numbers_cell.select('span.jqh')
        front_numbers = []
        for span in front_spans:
            if span.text.strip():
                num = int(span.text.strip())
                front_numbers.append(num)

        back_spans = back_numbers_cell.select('span.jql')
        back_numbers = []
        for span in back_spans:
            if span.text.strip():
                num = int(span.text.strip())
                back_numbers.append(num)

        front_numbers.sort()
        back_numbers.sort()
        front_numbers_str = ','.join(map(str, front_numbers))
        back_numbers_str = ','.join(map(str, back_numbers))

        total_sales = cells[4].text.strip().replace(',', '')
        first_prize_num = cells[5].text.strip()
        first_prize_money = cells[6].text.strip().replace(',', '')
        first_add_num = cells[7].text.strip()
        first_add_money = cells[8].text.strip().replace(',', '')
        second_prize_num = cells[9].text.strip()
        second_prize_money = cells[10].text.strip().replace(',', '')
        second_add_num = cells[11].text.strip()
        second_add_money = cells[12].text.strip().replace(',', '')
        prize_pool = cells[13].text.strip().replace(',', '')

        date_match = re.match(r'(\d{4}-\d{2}-\d{2})', date_str)
        if date_match:
            date_str = date_match.group(1)

        data.append({
            'issue_num': issue_num,
            'date': date_str,
            'front_numbers': front_numbers_str,
            'back_numbers': back_numbers_str,
            'total_sales': total_sales,
            'first_prize_num': first_prize_num,
            'first_prize_money': first_prize_money,
            'first_add_num': first_add_num,
            'first_add_money': first_add_money,
            'second_prize_num': second_prize_num,
            'second_prize_money': second_prize_money,
            'second_add_num': second_add_num,
            'second_add_money': second_add_money,
            'prize_pool': prize_pool
        })

    return data

webparser()