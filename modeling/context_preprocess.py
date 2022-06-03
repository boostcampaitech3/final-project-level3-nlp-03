# 필요한 객체 import
import pandas as pd
from konlpy.tag import Okt
from preprocessing import preprocess_data, tokenizing_data, get_label
from utils import seed_fix
import pandas as pd
import numpy as np
import random

import re
# 초기화 및 모델 학습
from gensim.models import word2vec
# 띄어쓰기
from pykospacing import Spacing
# 마춤뻡 검사기
from hanspell import spell_checker

from soynlp.normalizer import *

def preprocess_context(contexts):
    for word in contexts:
        if type(word) == float:
            contexts.remove(word)
    
    def remove_repeat_char(contexts):
        preprocessed_text = []
        for text in contexts:
            text = repeat_normalize(text, num_repeats=2).strip()
            if text:
                preprocessed_text.append(text)
        print("remove_repeat_char done")
        return preprocessed_text
    
    # 한국어 정규식 확인 + ㅏ, ㅋ 이런 거 제거
    def regular_check(contexts):
        preprocessed_text = []
        for text in contexts:
            text = re.sub('[ㄱ-ㅎㅏ-ㅣ]', '', text)
            text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', text)
            if text:
                preprocessed_text.append(text)
        print("regular_check done")
        return preprocessed_text
    
    # 특수문자 통일
    def clean_punc(texts):
        punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

        preprocessed_text = []
        for text in texts:
            for p in punct_mapping:
                text = text.replace(p, punct_mapping[p])
            text = text.strip()
            if text:
                preprocessed_text.append(text)
        print("clean_punc done")
        return preprocessed_text

    def remove_repeated_spacing(texts):
        """
        두 개 이상의 연속된 공백을 하나로 치환합니다.
        ``오늘은    날씨가   좋다.`` -> ``오늘은 날씨가 좋다.``
        """
        preprocessed_text = []
        for text in texts:
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                preprocessed_text.append(text)
        print("remove_repeated_spacing done")
        return preprocessed_text
    
    def spacing_sent(texts):
        """
        띄어쓰기를 보정합니다.
        """
        preprocessed_text = []
        spacing = Spacing()
        for text in texts:
            text = spacing(text)
            if text:
                preprocessed_text.append(text)
        print("spacing_sent done")
        return preprocessed_text
    
    def spell_check_sent(texts):
        """
        맞춤법을 보정합니다.
        """
        preprocessed_text = []
        for text in texts:
            try:
                spelled_sent = spell_checker.check(text)
                checked_sent = spelled_sent.checked 
                if checked_sent:
                    preprocessed_text.append(checked_sent)
            except:
                preprocessed_text.append(text)
        print("spell_check_sent done")
        return preprocessed_text
    
    contexts = remove_repeat_char(contexts)
    contexts = regular_check(contexts)
    contexts = clean_punc(contexts)
    contexts = remove_repeated_spacing(contexts)
    contexts = spacing_sent(contexts)
    contexts = spell_check_sent(contexts)

    return contexts