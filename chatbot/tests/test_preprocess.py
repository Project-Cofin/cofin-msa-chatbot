from chatbot.utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle
import pandas as pd


class PreTest:
    def __init__(self):
        pass

    def execute_sentence(self):
        # sent = '2022년 1월에 뮤지컬을 보러 가기로 했는데 뮤지컬도 백신을 맞아야만 들어갈 수 있는 건가요? 제가 듣기론 위드 코로나가 코로나 완치자와 19세 미만 청소년 등에게는 해당사항이 없는 걸로 알고 있는데 확실하게 알고 싶어서요'
        sent = '코로나 감염 질문 제가 지금 코로나에 걸린 상태인데 입대고 마신 음료수가 냉장고에 있거든요 근데 제가 코로나 완치한 후에 그 입대고 마셨던 음료수를 다시 입대고 마시면 다시 코로나에 걸리나요'

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
    pt.execute_sentence()
    # pt.create_wb()
    # pt.test_wb()