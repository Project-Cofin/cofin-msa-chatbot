import re

from common.models import ValueObject
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import csv


class TestC(object):
    def __init__(self):
        pass

    def execute_crawling(self):
        vo = ValueObject()
        vo.context = '../data/'
        vo.url = 'https://kin.naver.com/qna/detail.naver?d1id=7&dirId=7010105&docId=405847522&qb=7L2U66Gc64KY&enc=utf8&section=kin.qna.all&rank=1&search_sort=0&spq=0'
        # vo.url = 'https://kin.naver.com/qna/detail.naver?d1id=7&dirId=701&docId=404508069&qb=7L2U66Gc64KY&enc=utf8&section=kin&rank=8&search_sort=0&spq=0'
        driver = webdriver.Chrome(f'{vo.context}chromedriver')
        driver.get(vo.url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        question = soup.select_one('.question-content div.c-heading__content').get_text().strip()
        answers = soup.find_all('div', '_endContentsText c-heading-answer__content-user')
        print(f'질문: {question}')
        for answer in answers:
            print(f'답변: {answer.get_text().strip()}')
        # pd.to_csv(vo.context + '/test.csv', index=False)
        # print(df)
        driver.close()


if __name__ == '__main__':
    c = TestC()
    c.execute_crawling()