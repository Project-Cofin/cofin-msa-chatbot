from koalanlp.Util import initialize, finalize
from koalanlp.proc import *
from koalanlp import API
import os
import sys

class Blog:
    def __init__(self):
        self.ETRI_API_KEY = "36e9fdcf-32bf-4ded-a4df-ad257513b8e5" # ETRI에서 발급받은 키를 입력하세요.

    def execute(self):
        # 초기화 합니다.
        initialize(java_options="-Xmx4g -Dfile.encoding=utf-8", KKMA="2.0.2", EUNJEON="2.0.2", ETRI="2.0.2")

        # 품사분석기 이용법
        tagger = Tagger(API.EUNJEON)
        tagged = tagger.tag("안녕하세요. 눈이 오는 설날 아침입니다.")
        print(tagged)

        # 의존구문분석기 이용법
        parser = Parser(API.KKMA)
        parsed = parser.analyze("안녕하세요. 눈이 오는 설날 아침입니다.")
        print(parsed)

        # ETRI API 이용법
        rolelabeler = RoleLabeler(API.ETRI, etri_key=self.ETRI_API_KEY)
        paragraph = rolelabeler.analyze("첫 분석을 시도해봅시다!")
        print(paragraph)
        print(paragraph[0].getRoles())

        # Data classes
        sentence = parsed[1]  # 두번째 문장인, "눈이 오는 설날 아침입니다."를 선택합니다.

        wordAt0 = sentence[0]  # 첫번째 어절을 선택해봅니다.
        print(wordAt0.exists(lambda m: m.isPredicate()))  # 첫번째 어절에, 용언(동사/형용사)을 포함한 형태소가 있는지 확인합니다.
        print(sentence.exists(lambda w: w.exists(lambda m: m.isNoun())))  # 문장 전체에 체언(명사 등)을 포함한 어절이 있는지 확인합니다.
        print(sentence.getNouns())  # 문장에서 체언만 추출합니다.
        print(sentence.getVerbs())  # 문장에서 용언만 추출합니다.

        finalize()  # KoalaNLP 사용을 종료합니다.

    def etri_api_usage(self):
        initialize(ETRI='LATEST')
        labeler = RoleLabeler(API.ETRI, etri_key=self.ETRI_API_KEY)
        # recognizer = EntityRecognizer(etri_key=API_KEY)
        # parser = Parser(etri_key=API_KEY)
        # tagger = Tagger(etri_key=API_KEY)

        while True:
            text = input("분석할 문장을 입력하세요>> ").strip()

            if len(text) == 0:
                break

            sentences = labeler(text)

            for sent in sentences:
                print("===== Sentence =====")
                print(sent.singleLineString())

                entities = sent.getEntities()
                if len(entities) > 0:
                    print("# Named Entities")

                    for entity in entities:
                        print("[%s]는 %s 유형의 개체명으로, 형태소 [%s]를 포함합니다." % (entity.getSurface(),
                                                                        str(entity.getFineLabel()),
                                                                        " ".join(str(m) for m in entity)))

                dependencies = sent.getDependencies()
                if len(dependencies) > 0:
                    print("# Dependency Parse")

                    for edge in dependencies:
                        print("[%s]는 [%s]의 %s-%s" % (edge.getDependent().getSurface(),
                                                     edge.getGovernor().getSurface() if edge.getGovernor() is not None else "ROOT",
                                                     str(edge.getType()),
                                                     str(edge.getDepType())))

                roles = sent.getRoles()
                if len(roles) > 0:
                    print("# Role Labeling")

                    for edge in roles:
                        print("[%s]는 [%s]의 %s" % (edge.getArgument().getSurface(),
                                                  edge.getPredicate().getSurface(),
                                                  str(edge.getLabel())))

        finalize()

    def sentence_split_usage(self):
        initialize(OKT='LATEST')  #: HNN=2.0.3

        splitter = SentenceSplitter(API.OKT)

        while True:
            text = input("분석할 문장을 입력하세요>> ").strip()

            if len(text) == 0:
                break

            sentences = splitter(text)

            print("===== Sentence =====")
            for i, sent in enumerate(sentences):
                print("[%s] %s" % (i, sent))

        finalize()

    def dep_parser_usage(self):
        initialize(KKMA='LATEST')  #: HNN=2.0.4, ETRI=2.0.4

        parser = Parser(API.KKMA)

        while True:
            text = input("분석할 문장을 입력하세요>> ").strip()

            if len(text) == 0:
                break

            sentences = parser(text)

            for sent in sentences:
                print("===== Sentence =====")
                print(sent.singleLineString())
                print("# Dependency Parse result")

                dependencies = sent.getDependencies()
                if len(dependencies) > 0:
                    for edge in dependencies:
                        print("[%s]는 [%s]의 %s-%s" % (edge.getDependent().getSurface(),
                                                     edge.getGovernor().getSurface() if edge.getGovernor() is not None else "ROOT",
                                                     str(edge.getType()),
                                                     str(edge.getDepType())))
                else:
                    print("(Unexpected) NULL!")

        finalize()


if __name__ == '__main__':
    b = Blog()
    # b.execute()
    # b.etri_api_usage()
    # b.sentence_split_usage()
    b.dep_parser_usage()