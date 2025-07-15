import csv
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

expt = set()
expt_data = []

def webparser():
	global expt
	global expt_data
	opt = webdriver.ChromeOptions()
	opt.add_argument('--blink-settings=imagesEnabled=false')

	driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=opt)

	driver.get("https://www.cmzj.net/dlt/tickets")

	# 等待菜单加载
	period_tabs = WebDriverWait(driver, 10).until(
		EC.presence_of_element_located((By.CSS_SELECTOR, ".rk-main .head > ul.el-menu-demo"))
	)

	# 依次点击四个榜单
	tab_names = ["本期", "周榜", "月榜", "年榜"]
	for tab_name in tab_names:
		js_script = f"""
			var tabs = document.querySelectorAll('.rk-main .head > ul.el-menu-demo li');
			for(var i = 0; i < tabs.length; i++) {{
				if(tabs[i].textContent.trim() === '{tab_name}') {{
					tabs[i].click();
					return true;
				}}
			}}
			return false;
		"""
		success = driver.execute_script(js_script)

		WebDriverWait(driver, 5).until(
			EC.presence_of_element_located((By.CSS_SELECTOR, ".content table tbody tr"))
		)

		experts = driver.find_elements(By.CSS_SELECTOR, ".content table tbody tr")

		for index, expert in enumerate(experts, 1):
			if index > 10:
				break
			history_td = expert.find_elements(By.TAG_NAME, "td")[4]

			original_window = driver.current_window_handle

			# 使用JavaScript执行点击
			driver.execute_script("arguments[0].click();", history_td)

			# 等待新窗口出现
			WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))

			# 切换到新窗口
			for window_handle in driver.window_handles:
				if window_handle != original_window:
					driver.switch_to.window(window_handle)
					break

			cururl = driver.current_url

			if cururl in expt:
				driver.close()
				driver.switch_to.window(original_window)
				WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.items")))
				time.sleep(2)
				continue
			else:
				expt.add(cururl)

			time.sleep(2)

			page_source = driver.page_source
			soup = BeautifulSoup(page_source, 'lxml')

			expertparser(soup)

			driver.close()
			driver.switch_to.window(original_window)
			WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.items")))

			time.sleep(0.5)

			if len(expt) > 20:
				exit

	while len(expt) <= 20:
		pageparser(driver)

		button = WebDriverWait(driver, 10).until(
			EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'change') and .//p[text()='换一批']]"))
		)
		driver.execute_script("arguments[0].click();", button)

	driver.quit()

	print(expt_data)
	export_to_csv(expt_data)

def pageparser(driver):
	global expt

	WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
		(By.CSS_SELECTOR, "div.items")
	))

	for index in range(8):
		expert_list = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
			(By.CSS_SELECTOR, "div.items")
		))

		expert = expert_list[index]

		name_element = expert.find_element(By.CSS_SELECTOR, "p.okami-name")

		driver.execute_script("arguments[0].scrollIntoView();", name_element)
		time.sleep(0.5)

		original_window = driver.current_window_handle

		driver.execute_script("arguments[0].click();", name_element)

		WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))
		for window_handle in driver.window_handles:
			if window_handle != original_window:
				driver.switch_to.window(window_handle)
				break

		cururl = driver.current_url

		# print(cururl)

		if cururl in expt:
			driver.close()
			driver.switch_to.window(original_window)
			WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.items")))
			time.sleep(2)
			continue
		else:
			expt.add(cururl)

		time.sleep(2)

		page_source = driver.page_source
		soup = BeautifulSoup(page_source, 'lxml')

		expertparser(soup)

		driver.close()
		driver.switch_to.window(original_window)
		WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.items")))
		time.sleep(2)

def expertparser(soup):
	global expt_data

	lottery_exp_element = soup.select_one('.okami-text p:contains("彩龄：") span')
	lottery_experience = lottery_exp_element.text

	post_count_element = soup.select_one('.okami-text p:contains("文章数量：") span')
	post_count = post_count_element.text

	dlt_data = {"一等奖": 0, "二等奖": 0, "三等奖": 0}

	for djzj_div in soup.select('.djzj'):
		if djzj_div.find('span', class_='text-head-bg', text='大乐透'):
			for item in djzj_div.select('.item'):
					text = item.get_text(strip=True)
					match = re.match(r'(.*?)(\d+)次', text)

					award_type = match.group(1)
					count = match.group(2)

					dlt_data[award_type] = count

	expt_data.append({
		"彩龄": lottery_experience,
		"发文量": post_count,
		"大乐透中奖情况": dlt_data
	})

def export_to_csv(data, filename='../expert_data.csv'):
	processed_data = []

	for item in data:
		processed_item = {}

		processed_item['彩龄'] = int(item['彩龄'].replace('年', ''))

		processed_item['发文量'] = int(item['发文量'].replace('篇', ''))

		processed_item['一等奖'] = int(item['大乐透中奖情况']['一等奖'])
		processed_item['二等奖'] = int(item['大乐透中奖情况']['二等奖'])
		processed_item['三等奖'] = int(item['大乐透中奖情况']['三等奖'])

		processed_data.append(processed_item)

	df = pd.DataFrame(processed_data)
	df.to_csv(filename, index=False, encoding='utf-8-sig')

webparser()