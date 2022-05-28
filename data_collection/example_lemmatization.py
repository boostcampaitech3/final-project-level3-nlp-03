# from konlpy.tag import Twitter
from konlpy.tag import Okt
from konlpy.tag import Kkma
from hanspell import spell_checker
# from pykospacing import Spacing


def lemmatization(text):
    okt = Okt()
    kkma = Kkma()
    hanspell_result = spell_checker.check(text)
    text = hanspell_result.checked
    
    print(text)
    print(okt.morphs(text))
    print(kkma.morphs(text))

    # spacing = Spacing()
    # kospacing_text = spacing(text) 


def main():
    text1 = "날아왔습니다 날아왔어요 날아왓음 날아옴"
    text2 = "TV,냉장고등에게 전류가모두가고있기때문" #q1 40번 answer
    # lemmatization(text1)
    lemmatization(text2)
    lemmatization(text3)


if __name__ == "__main__":
    main()