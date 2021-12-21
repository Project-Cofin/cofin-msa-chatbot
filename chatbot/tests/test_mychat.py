from gensim.models import Word2Vec
from konlpy.tag import Komoran
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate


class MyChat:
    def __init__(self):
        pass

    def execute(self):
        # 측정 시작
        start = time.time()

        # 리뷰 파일 읽어오기
        print('1) 말뭉치 데이터 읽기 시작')
        review_data = pd.read_csv('../data/sample_chat.csv')
        review_data = review_data['question']
        print(len(review_data)) # 리뷰 데이터 전체 개수
        print('1) 말뭉치 데이터 읽기 완료: ', time.time() - start)

        # 문장단위로 명사만 추출해 학습 입력 데이터로 만듬
        print('2) 형태소에서 명사만 추출 시작')
        komoran = Komoran(userdic='../data/user_dic.tsv')
        docs = [komoran.nouns(sentence) for sentence in review_data]
        print('2) 형태소에서 명사만 추출 완료: ', time.time() - start)
        print(review_data[0])
        print(komoran.pos(review_data[0]))
        print(docs[0])
        # word2vec 모델 학습
        # print('3) word2vec 모델 학습 시작')
        # model = Word2Vec(sentences=docs, vector_size=200, window=4, min_count=2, sg=1)
        # print('3) word2vec 모델 학습 완료: ', time.time() - start)

        # 모델 저장
        # print('4) 학습된 모델 저장 시작')
        # model.save('../data/new_data/nvmc.model')
        # print('4) 학습된 모델 저장 완료: ', time.time() - start)

        # 학습된 말뭉치 개수, 코퍼스 내 전체 단어 개수
        # print("corpus_count : ", model.corpus_count)
        # print("corpus_total_words : ", model.corpus_total_words)

    def execute2(self):
        model = Word2Vec.load('../data/new_data/nvmc.model')
        print(f"corpus 전체 단어 갯수: {model.corpus_total_words}")
        print(f"'증상'란 단어로 생성한 단어 임베딩 벡터: {model.wv['증상']}")
        print(f"'증상'란 단어와 유사한 단어들: {model.wv.most_similar('증상')}")

    def execute3(self):
        # 데이터 읽어오기
        train_file = "../data/sample_chat.csv"
        data = pd.read_csv(train_file)
        features = data['question'].tolist()
        labels = data['label'].tolist()

        # 단어 인덱스 시퀀스 벡터
        corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]

        tokenizer = preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(corpus)
        sequences = tokenizer.texts_to_sequences(corpus)
        word_index = tokenizer.word_index
        MAX_SEQ_LEN = 57  # 단어 시퀀스 벡터 크기
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

        # 학습용, 검증용, 테스트용 데이터셋 생성 ➌
        # 학습셋:검증셋:테스트셋 = 7:2:1
        ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
        ds = ds.shuffle(len(features))
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
        VOCAB_SIZE = len(word_index) + 1  # 전체 단어 수

        # CNN 모델 정의
        input_layer = Input(shape=(MAX_SEQ_LEN,))
        embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
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
        model.save('../data/cnn_model.h5')

    def execute_predict(self):
        # 데이터 읽어오기
        train_file = "../data/sample_chat.csv"
        data = pd.read_csv(train_file, delimiter=',')
        features = data['question'].tolist()
        labels = data['label'].tolist()

        # 단어 인덱스 시퀀스 벡터
        corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
        tokenizer = preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(corpus)
        sequences = tokenizer.texts_to_sequences(corpus)
        MAX_SEQ_LEN = 57  # 단어 시퀀스 벡터 크기
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

        # 테스트용 데이터셋 생성
        ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
        ds = ds.shuffle(len(features))
        test_ds = ds.take(2000).batch(20)  # 테스트 데이터셋

        # 감정 분류 CNN 모델 불러오기
        model = load_model('../data/cnn_model.h5')
        model.summary()
        model.evaluate(test_ds, verbose=2)

        # 테스트용 데이터셋의 0번째 데이터 출력
        # print("단어 시퀀스 : ", corpus[0])
        # print("단어 인덱스 시퀀스 : ", padded_seqs[0])
        print("문장 분류(정답) : ", labels[88])

        # 테스트용 데이터셋의 0번째 데이터 의도 예측
        picks = [88]
        predict = model.predict(padded_seqs[picks])
        predict_class = tf.math.argmax(predict, axis=1)
        # print("의도 예측 점수 : ", predict)
        print("의도 예측 클래스 : ", predict_class.numpy())

if __name__ == '__main__':
    mc = MyChat()
    # mc.execute()
    # mc.execute2()
    # mc.execute3()
    mc.execute_predict()