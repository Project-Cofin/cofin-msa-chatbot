from konlpy.tag import Komoran
import numpy as np


class TestKobert(object):
    def __init__(self):
        pass

    def execute(self):
        komoran = Komoran(userdic='data/user_dic.tsv')
        text = "안녕하세요. 어제 운동 후 찬곳에 오래 있었더니 오한이 들기 시작하면서 열이 엄청 많이 났어요"\
               "다음날 아침에는 열+근육통+오한이 있어서 병원에 갔더니 열이 39도가 넘었어요!"\
                                 "병원에서 일단 약을 먹고 효과가 없으면 코로나 검사를 진행하고 음성판정을 받으면"\
                                 "병원내에서 수액을 맞아야 한다했는데"\
                                 "약을 받자마자 하나 먹었더니 열이내리고 땀이 나면서 컨디션이 갑자기 좋아졌어요!"\
                                 "이런 경우 코로나일 가능성이 높나요? 약으로 치료가 되어도 의무적으로 검사를 받아야 하는지 궁금합니다."
        nouns = komoran.nouns(text)
        print(nouns)

        # 단어 사전 구축 및 단어별 인덱스 부여
        dics = {}
        for word in nouns:
            if word not in dics.keys():
                dics[word] = len(dics)
        print(dics)

        # 원-핫 인코딩
        nb_classes = len(dics)
        targets = list(dics.values())
        one_hot_targets = np.eye(nb_classes)[targets]
        print(one_hot_targets)
        # text = "우리 챗봇은 엔엘피를 좋아해."
        # pos = komoran.pos(text)
        # print(pos)


if __name__ == '__main__':
    t = TestKobert()
    t.execute()