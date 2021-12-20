import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
from chatbot.utils.Preprocess import Preprocess
from chatbot.utils.IntentModel import IntentModel

class IntentChat:
    def __init__(self):
        self.MAX_SEQ_LEN = 57

    def createModel(self):
        # 데이터 읽어오기
        train_file = "../data/sample_chat.csv"
        data = pd.read_csv(train_file, delimiter=',')
        queries = data['question'].tolist()
        intents = data['label'].tolist()
        p = Preprocess(word2index_dic='../data/chatbot_dict.bin', userdic='../data/user_dic.tsv')

        sequences = []
        for sentence in queries:
            pos = p.pos(sentence)
            keywords = p.get_keywords(pos, without_tag=True)
            seq = p.get_wordidx_sequence(keywords)
            sequences.append(seq)

        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=self.MAX_SEQ_LEN, padding='post')
        ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))
        ds = ds.shuffle(len(queries))

        train_size = int(len(padded_seqs) * 0.7)
        val_size = int(len(padded_seqs) * 0.2)
        test_size = int(len(padded_seqs) * 0.1)

        train_ds = ds.take(train_size).batch(20)
        val_ds = ds.skip(train_size).take(val_size).batch(20)
        test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

        # 하이퍼파라미터 설정
        dropout_prob = 0.5
        EMB_SIZE = 128
        EPOCH = 50
        VOCAB_SIZE = len(p.word_index) + 1  # 전체 단어 수

        input_layer = Input(shape=(self.MAX_SEQ_LEN,))
        embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=self.MAX_SEQ_LEN)(input_layer)
        dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

        conv1 = Conv1D(filters=128, kernel_size=3, padding='valid', activation=tf.nn.relu)(dropout_emb)
        pool1 = GlobalMaxPool1D()(conv1)
        conv2 = Conv1D(filters=128, kernel_size=4, padding='valid', activation=tf.nn.relu)(dropout_emb)
        pool2 = GlobalMaxPool1D()(conv2)
        conv3 = Conv1D(filters=128, kernel_size=5, padding='valid', activation=tf.nn.relu)(dropout_emb)
        pool3 = GlobalMaxPool1D()(conv3)

        # 3, 4, 5- gram 이후 합치기
        concat = concatenate([pool1, pool2, pool3])
        hidden = Dense(128, activation=tf.nn.relu)(concat)
        dropout_hidden = Dropout(rate=dropout_prob)(hidden)
        logits = Dense(85, name='logits')(dropout_hidden)
        predictions = Dense(85, activation=tf.nn.softmax)(logits)

        # 모델 생성
        model = Model(inputs=input_layer, outputs=predictions)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 모델 학습
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

        # 모델 평가(테스트 데이터셋 이용)
        loss, accuracy = model.evaluate(test_ds, verbose=1)
        print('Accuracy: %f' % (accuracy * 100))
        print('loss: %f' % (loss))

        # 모델 저장
        model.save('../data/intent_model.h5')

    def predictModel(self):
        p = Preprocess(word2index_dic='../data/chatbot_dict.bin', userdic='../data/user_dic.tsv')

        intent = IntentModel(model_name='../data/intent_model.h5', proprocess=p)

        query = "미열하고 약간의 기침이 있어요... 코로나일까요? " \
               "배도 고파요, 저녁 메뉴는 뭘까요? 프로젝트는 잘 마칠수 있겠죠?"
        predict = intent.predict_class(query)
        predict_label = intent.labels[predict]

        print(query)
        print(f'의도 예측 클래스: {predict}')
        print(f'의도 예측 레이블: {predict_label}')


if __name__ == '__main__':
    ic = IntentChat()
    ic.createModel()
    # ic.predictModel()