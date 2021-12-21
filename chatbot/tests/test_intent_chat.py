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

        # query = "미열하고 약간의 기침이 있어요... 코로나일까요? " \
        #        "배도 고파요, 저녁 메뉴는 뭘까요? 프로젝트는 잘 마칠수 있겠죠?"
        # query = '아침부터 미열하고 인후통이 있어요. 혹시 코로나에 감염된건 아니겠죠?'
        # query = '어제 제가 있던 건물에서 확진자가 나와서 코로나 검사를 받았어요. 저도 자가격리 대상자인가요?'
        # query = '95년생은 백신 접종 언제부터 예약이 가능한가요?'
        # query = '아침에는 37도 정도 나왔는데, 오후되니까 38.3도까지 나와요.. 머리도 아프고 기침도 자꾸 나오는데 코로나 증상일까요..'
        # query = '위드코로나때도 굳이 마스크를 착용해야하나요?'
        # query = '백신패스는 어떻게 발급받는거죠? 접종은 안했구, 확진됐다가 완치만 됐는데 대상자인가요?'
        query = '나 머리가 아픈데 이게 뭘까 코로나일까?'

        predict = intent.predict_class(query)
        predict_label = intent.labels[predict]

        print(query)
        print(f'의도 예측 클래스: {predict}')
        print(f'의도 예측 레이블: {predict_label}')


if __name__ == '__main__':
    ic = IntentChat()
    # ic.createModel()
    ic.predictModel()