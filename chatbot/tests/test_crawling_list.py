import re

from common.models import ValueObject
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import csv


class TestC(object):
    def __init__(self):
        pass

    def execute(self):
        vo = ValueObject()
        vo.context = '../data/'
        q_list = []
        a_list = []
        for i in range(1, 990, 10):
            vo.url = f'https://search.naver.com/search.naver?where=kin&kin_display=10&qt=&title=0&&answer=0&grade=0&choice=0&sec=0&nso=so%3Ar%2Ca%3Aall%2Cp%3Aall&' \
                     f'query={"코로나"}&c_id=&c_name=&sm=tab_pge&' \
                     f'kin_start={i}&kin_age=0'
            driver = webdriver.Chrome(f'{vo.context}chromedriver')
            driver.get(vo.url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            questions = soup.find_all('div', 'question_group')
            answers = soup.find_all('div', 'answer_group')
            driver.close()
            for question, answer in zip(questions, answers):
                q_list.append(",".join([i for i in question.get_text().strip().split("...") if i]))
                a_list.append(",".join([i for i in answer.get_text().strip().split("...") if i]))
            # print(f'질문: { ",".join([i for i in question.get_text().strip().split("...") if i]) }')
            # print(f'답변: { ",".join([i for i in answer.get_text().strip().split("...") if i]) }')
        df = pd.DataFrame({'question': q_list, 'answer': a_list, 'intent': ''})
        # print(df)
        df.to_csv(f'{vo.context}dataset.csv', index=False)


if __name__ == '__main__':
    c = TestC()
    # c.execute()