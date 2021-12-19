import csv

from chatbot.models import HealthStatus
from common.models import ValueObject, Reader


class DbUploader():
    def __init__(self):
        self.vo = ValueObject()
        self.reader = Reader()
        self.vo.context = 'chatbot/data/'

    def insert_data(self):
        # self.insert_health_status()
        pass

    def insert_health_status(self):
        self.vo.fname = 'health_status.csv'
        self.csvfile = self.reader.new_file(self.vo)
        with open(self.csvfile, newline='', encoding='utf8') as csvfile:
            data_reader = csv.DictReader(csvfile)
            for row in data_reader:
                HealthStatus.objects.create(symptom=row['symptom'],
                                            details=row['details'],
                                            level=row['level'],
                                            answer=row['answer'])
            print('HealthStatus DATA UPLOADED SUCCESSFULLY!')