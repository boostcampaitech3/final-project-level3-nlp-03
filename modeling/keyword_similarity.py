# 필요한 객체 import
import pandas as pd
from konlpy.tag import Okt
from preprocessing import preprocess_data, tokenizing_data, get_label
from utils import seed_fix
import pandas as pd
import numpy as np
import random

import re

# 띄어쓰기
from pykospacing import Spacing
# 마춤뻡 검사기
from hanspell import spell_checker

from soynlp.normalizer import *

import gensim 
import gensim.models as g

model_name = 'fast_text_ko'
model = g.Doc2Vec.load(model_name)

import re

class Keyword_similarity:
    
    def __init__(self, model, threshold = 0.5, pos_tagger=None, lemmatizer=None):
        self.model = model
        self.threshold = threshold
        self.pos_tagger = pos_tagger
        self.lemmatizer = lemmatizer
    
    ### 키워드랑 문장이 리스트로 들어온다.
    def tokenize(self, doc):
    # norm, stem은 optional
        if type(doc) is not str:
            return []
        return ['/'.join(t) for t in self.pos_tagger.pos(doc, norm=True, stem=True)]

    def keyword_score(self, keyword_list, sentence_list):
        idx_list = []
        for keyword in keyword_list:
            idx_list.append(self.get_keyword_score_list(keyword, sentence_list))

        return idx_list

    ## 키워드 하나에 대해서 답안들에 대해 조사한다.
    def get_keyword_score_list(self, keyword, sentence_list):
        idx_list = []
        for sentence in sentence_list:
            idx_list.append(self.keyword_one_sentence(keyword, sentence))
        
        return idx_list

    ## 하나의 키워드에 대해서 그 문장의 것들에 대한 점수를 리스트로 반환한다.
    def keyword_one_sentence(self, keyword, sentence):
        pos_keyword = self.tokenize(keyword)
        keyword_vec = self.model.wv.get_vector(pos_keyword[0])
        
        cosine_list = []
        word_list = []
        for word in sentence:
            
            pos_word = self.tokenize(word)
            try: 
                word_vec = self.model.wv.get_vector(pos_word[0])
            except:
                continue
            if cosine_similarity(keyword_vec, word_vec) > self.threshold and pos_word not in word_list:
                results = re.finditer(word, sentence)
                for matched_key in results:
                    cosine_list.append([matched_key.start(), matched_key.end()])
                
                word_list.append(pos_word)
            
        return cosine_list

def main(keyword_list, sentence_list, KS):
    
    return KS.keyword_score(keyword_list, sentence_list)

if __name__ == '__main__':
    # 파일 불러오기
    sentence_data_path = args.sentence_data_path
    sentence_file_name = '../data/' + args.sentence_file_name
    data = pd.read_csv(sentence_file_name, encoding='utf-8')
    sentence_list = data['sentence'].tolist()

    keyword_data_path = args.keyword_data_path
    keyword_file_name = '../data/' + args.keyword_file_name
    data = pd.read_csv(keyword_file_name, encoding='utf-8')
    keyword_list = data['keyword'].tolist()

    # 모델 불러오기

    model_name = args.model_name
    model = g.Doc2Vec.load(model_name)
    Fast_KS = Keyword_similarity(model, 0.35, pos_tagger)
    main(sentence_list, keyword_list, Fast_KS)
