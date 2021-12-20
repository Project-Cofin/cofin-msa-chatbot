import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing


class IntentModel:
    def __init__(self):
        # self.labels = lambda x: {x: '1'} in range(5)
        self.labels = {x: '1' for x in range(5)}
        self.intents = ['증상 질문', '증상 순서', '경미한 의심증상', '주의단계 의심증상', '치명적인 의심정황', '밀접 접촉자 증상 질문']

        print(self.labels)


if __name__ == '__main__':
    im = IntentModel()