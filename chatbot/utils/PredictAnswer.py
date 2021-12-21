from chatbot.utils.Preprocess import Preprocess
from chatbot.utils.IntentModel import IntentModel


class IntentChat:
    def __init__(self):
        pass

    def predictModel(self, query):
        p = Preprocess(word2index_dic='chatbot/data/chatbot_dict.bin', userdic='chatbot/data/user_dic.tsv')
        intent = IntentModel(model_name='chatbot/data/intent_model.h5', proprocess=p)
        return intent.predict_class(query)
