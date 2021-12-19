import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf


# 예제 참고: https://wikidocs.net/89786
class TransformerChat:
    def __init__(self):
        self.QUESTION_MAX_LENGTH = 55
        self.INTENT_MAX_LENGTH = 10
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 20000
        # 하이퍼 파라미터
        self.D_MODEL = 256
        self.NUM_LAYERS = 2
        self.NUM_HEADS = 8
        self.DFF = 512
        self.DROPOUT = 0.1
        self.EPOCHS = 50

    def execute(self):
        train_data = pd.read_csv('../data/sample_chat.csv')
        # print(train_data.head())
        # print('챗봇 샘플의 개수 :', len(train_data))
        # print(train_data.isnull().sum())
        questions = []
        for sentence in train_data['question']:
            # 구두점에 대해서 띄어쓰기
            # ex) 12시 땡! -> 12시 땡 !
            sentence = re.sub(r"([+#?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            questions.append(sentence)
        # print(questions[:5])
        intents = []
        for sentence in train_data['intent']:
            # 구두점에 대해서 띄어쓰기
            # ex) 12시 땡! -> 12시 땡 !
            sentence = re.sub(r"([+#?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            intents.append(sentence)

        # ------------- 단어 집합 생성 -------------
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            questions + intents, target_vocab_size=2 ** 13)
        START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
        VOCAB_SIZE = tokenizer.vocab_size + 2
        print(tokenizer)
        print('시작 토큰 번호 :', START_TOKEN)
        print('종료 토큰 번호 :', END_TOKEN)
        print('단어 집합의 크기 :', VOCAB_SIZE)
        # tokenized_string = tokenizer.encode(questions[58])
        tokenized_string = tokenizer.encode(intents[58])
        print(f'인코딩 샘플 길이 : {len(tokenized_string)}')
        print(f'임의의 질문 샘플을 정수 인코딩 : {tokenized_string}')
        original_string = tokenizer.decode(tokenized_string)
        print(f'기존 문장: {original_string}')

        # ------------- 단어별 토큰값 확인 -------------
        # for ts in tokenized_string:
        #     print(f'{ts} ----> {tokenizer.decode([ts])}')

        # ------------- 토큰화 된 문자배열 생성 -------------
        tokenized_inputs, tokenized_outputs = [], []
        for (sentence1, sentence2) in zip(questions, intents):
            # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
            sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

        # ------------- 패딩 -------------
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.QUESTION_MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.INTENT_MAX_LENGTH, padding='post')
        questions, intents = tokenized_inputs, tokenized_outputs
        # print(intents)
        print(f'질문 데이터의 크기(shape) : {questions.shape}')
        print(f'답변 데이터의 크기(shape) : {intents.shape}')
        # print(intents[58])

        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': questions,
                'dec_inputs': intents[:, :-1]  # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
            },
            {
                'outputs': intents[:, 1:]  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
            },
        ))
        dataset = dataset.cache()
        dataset = dataset.shuffle(self.BUFFER_SIZE)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print(f'기존 샘플: {intents[58]}')
        print(f'마지막 패딩 토큰을 제거한 샘플: {intents[58:59][:, :-1]}')  # 길이가 9가 된다.
        print(f'처음 패딩 토큰을 제거한 샘플: {intents[58:59][:, 1:]}')    # 길이는 역시 9가 된다.
        print(f'처음과 마지막 패딩 토큰을 제거한 샘플: {intents[58:59][:, 1:-1]}')    # 길이는 8이 된다.

        # ------------- 트랜스포머 생성 -------------
        # tf.keras.backend.clear_session()
        # model = transformer(
        #     vocab_size=VOCAB_SIZE,
        #     num_layers=self.NUM_LAYERS,
        #     dff=self.DFF,
        #     d_model=self.D_MODEL,
        #     num_heads=self.NUM_HEADS,
        #     dropout=self.DROPOUT)
        # learning_rate = 0.001
        #
        # optimizer = tf.keras.optimizers.Adam(
        #     learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        #
        # def accuracy(y_true, y_pred):
        #     # 레이블의 크기는 (batch_size, MAX_LENGTH - 1)
        #     y_true = tf.reshape(y_true, shape=(-1, self.INTENT_MAX_LENGTH - 1))
        #     return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        #
        # model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=[accuracy])
        # model.fit(dataset, epochs=self.EPOCHS)

#
# def transformer(vocab_size, num_layers, dff,
#                 d_model, num_heads, dropout,
#                 name="transformer"):
#
#       # 인코더의 입력
#       inputs = tf.keras.Input(shape=(None,), name="inputs")
#
#       # 디코더의 입력
#       dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
#
#       # 인코더의 패딩 마스크
#       enc_padding_mask = tf.keras.layers.Lambda(
#           create_padding_mask, output_shape=(1, 1, None),
#           name='enc_padding_mask')(inputs)
#
#       # 디코더의 룩어헤드 마스크(첫번째 서브층)
#       look_ahead_mask = tf.keras.layers.Lambda(
#           create_look_ahead_mask, output_shape=(1, None, None),
#           name='look_ahead_mask')(dec_inputs)
#
#       # 디코더의 패딩 마스크(두번째 서브층)
#       dec_padding_mask = tf.keras.layers.Lambda(
#           create_padding_mask, output_shape=(1, 1, None),
#           name='dec_padding_mask')(inputs)
#
#       # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
#       enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
#           d_model=d_model, num_heads=num_heads, dropout=dropout,
#       )(inputs=[inputs, enc_padding_mask]) # 인코더의 입력은 입력 문장과 패딩 마스크
#
#       # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
#       dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
#           d_model=d_model, num_heads=num_heads, dropout=dropout,
#       )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
#
#       # 다음 단어 예측을 위한 출력층
#       outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
#
#       return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)



if __name__ == '__main__':
    t = TransformerChat()
    t.execute()
