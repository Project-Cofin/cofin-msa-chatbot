import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing


class IntentModel:
    def __init__(self, model_name, proprocess):
        self.intents = ['증상 질문', '증상 순서', '경미한 의심증상', '주의단계 의심증상', '치명적인 의심정황',
                        '밀접 접촉자 증상 질문', '경미한 외부접촉', '접종 간격', '확진자 접촉 경미', '코로나 검사 의무',
                        '코로나 검사 타지역', '코로나 검사 가능 여부', '코로나 검사 위치', '코로나 검사 비용', '코로나 치료 비용',
                        '코로나 검사 기간', '코로나 종식 기간', '코로나 종식 국가', '코로나 잠복기', '코로나 정상참작',
                        '코로나 격리실', '코로나 사망', '코로나 서적', '코로나 생존기간', '코로나 해열제',
                        '잠복기간 검사', '위드코로나', '위드코로나 검사 시행', '위드코로나 마스크', '위드코로나 단계',
                        '위드코로나 정상 영업', '위드코로나 장단점', '위드코로나 중단 조건', '위드코로나 여부', '위드코로나 등교',
                        '위드코로나 자가격리', '위드코로나 종교시설', '위드코로나 상세', '비염 비교', '입병 비교',
                        '편도염 비교', '인후통 비교', '냉방병 비교', '피부질환 비교', '위드코로나 군휴가',
                        '코로나 군생활', '코로나 입소', '여행시 검사', '항공권 요금', '항공권 요금 미국',
                        '검사 비용 태국', '본인 확진으로 인한 주변여파', '진단키트 음성', '부스터샷 대상', '검사 결과 수령',
                        '검사 과정 질문', '완치자 백신 패스', '완치자 접종', '접종완료자 감염', '접종 건강상태',
                        '학생 접종시기', '확진자 실비', '확진자 접종', '확진자 동거', '확진자 자가격리',
                        '실내 체육시설', '밀접접촉자 음성', '자가격리 수칙', '백신생성 질문', '미성년자 접종',
                        '백신접종 시기', '백신접종 부작용', '환자 백신접종', '백신접종 하혈', '대학병원 검사',
                        '접종 증명서', '접종 후 검사', '검사 후 접종', '백신접종 장점', '백신패스',
                        '1차접종 백신패스', '백신 예약 확인', '상생지원금', '욕설', '알수없음']
        self.labels = {i: label for i, label in enumerate(self.intents)}
        self.model = load_model(model_name)
        self.p = proprocess
        self.MAX_SEQ_LEN = 57

    def predict_class(self, query):
        pos = self.p.pos(query)

        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=self.MAX_SEQ_LEN, padding='post')
        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]
