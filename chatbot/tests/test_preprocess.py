from chatbot.utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle
import pandas as pd


class PreTest:
    def __init__(self):
        pass

    def execute_sentence(self):
        sent = '코로나 37.2도 모양은 아래처럼 생겼고 체온을 잴때마다 체온이 달라지고 거의 37도 이상 37.5도 미만으로 나오는데 학교 같은반에 코로나가 있어서 자가격리 중이거든요 ㅠㅠ 너무 불안해요 36도 나올때도 있고 손목재니까 35.6나오고요 ㅠㅠㅠ 저 코로나 일까요..?'

        p = Preprocess(userdic='../data/user_dic.tsv')

        pos = p.pos(sent)

        ret = p.get_keywords(pos, without_tag=False)
        print(ret)

        ret = p.get_keywords(pos, without_tag=True)
        print(ret)

    def create_wb(self):
        corpus_data = pd.read_csv('../data/sample_chat.csv')
        corpus_data = corpus_data['question']
        p = Preprocess(userdic='../data/user_dic.tsv')
        dict = []
        for c in corpus_data:
            pos = p.pos(c)
            dict.append(p.get_keywords(pos, without_tag=True))
            # for k in pos:
            #     dict.append(k[0])
        # print(dict)
        tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
        tokenizer.fit_on_texts(dict)
        word_index = tokenizer.word_index

        f = open("../data/chatbot_dict.bin", "wb")
        try:
            pickle.dump(word_index, f)
        except Exception as e:
            print(e)
        finally:
            f.close()

    def test_wb(self):
        f = open("../data/chatbot_dict.bin", "rb")
        word_index = pickle.load(f)
        f.close()

        sent = "미열하고 약간의 기침이 있어요... 코로나일까요? " \
               "배도 고파요, 저녁 메뉴는 뭘까요? 프로젝트는 잘 마칠수 있겠죠?"

        p = Preprocess(userdic='../data/user_dic.tsv')
        pos = p.pos(sent)

        keywords = p.get_keywords(pos, without_tag=True)
        for word in keywords:
            try:
                print(word, word_index[word])
            except KeyError:
                print(word, word_index['OOV'])

if __name__ == '__main__':
    pt = PreTest()
    # pt.execute_sentence()
    # pt.create_wb()
    pt.test_wb()