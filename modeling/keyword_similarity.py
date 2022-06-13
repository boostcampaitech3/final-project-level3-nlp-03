# 필요한 객체 import
from ast import keyword
import pandas as pd
from konlpy.tag import Okt
from preprocessing import preprocess_data, tokenizing_data, get_label
from utils import seed_fix
import pandas as pd
import numpy as np
import random

from context_preprocess import preprocess_context

import re

# 띄어쓰기
from pykospacing import Spacing
# 마춤뻡 검사기
from hanspell import spell_checker

from soynlp.normalizer import *

import gensim 
import gensim.models as g

import re

from keyword_arguments import get_args

def cosine_similarity(a, b):

    #return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


class Keyword_similarity:
    
    def __init__(self, model, threshold = 0.5, pos_tagger=None, lemmatizer=None):
        self.model = model
        self.threshold = threshold
        self.pos_tagger = pos_tagger
        self.lemmatizer = lemmatizer
    
    ### 키워드랑 문장이 리스트로 들어온다.
    # def tokenize(self, doc):
    # # norm, stem은 optional
    #     if type(doc) is not str:
    #         return []
    #     return ['/'.join(t) for t in self.pos_tagger.pos(doc, norm=True, stem=True)]
    
    def get_keyword_score(self, keyword_list, sentence_list):
        result = self.keywords_sentences(keyword_list, sentence_list)
        
        keyword_num = len(keyword_list)
        sentence_num = len(sentence_list)
        corrected = []
        for keyword, students in result.items():
            temp = []
            for idx, sentence in students.items():    
                if len(sentence[0]["start_idx"]) != 0:
                    temp.append(1)
                else : 
                    temp.append(0)

            corrected.append(temp)

        keyword_score = [0 for i in range(sentence_num)]
        for score in corrected:
            for i in range(sentence_num):
                keyword_score[i] += score[i]

        # print(keyword_score)
        # print(result)

        return keyword_score, result
        

    def keywords_sentences(self, keyword_list, sentence_list):
        keyword_dict = {}
        for idx, keyword in enumerate(keyword_list):
            keyword_dict[f"keyword_{idx}"] = self.keyword_sentences(keyword, sentence_list)

        return keyword_dict

    ## 키워드 하나에 대해서 답안들에 대해 조사한다.
    def keyword_sentences(self, keyword, sentence_list):
        sentence_dict = {}
        for idx, sentence in enumerate(sentence_list):
            sentence_dict[f'student_{idx}'] = self.keyword_one_sentence(keyword, sentence)
        
        return sentence_dict

    ## 하나의 키워드에 대해서 그 문장의 것들에 대한 점수를 리스트로 반환한다.
    def keyword_one_sentence(self, keyword, sentence):
        # pos_keyword = self.tokenize(keyword)
        # keyword_vec = self.model.wv.get_vector(pos_keyword[0])
        pos_keyword = self.pos_tagger.pos(keyword)
        keyword_vec = self.model.wv.get_vector(pos_keyword[0][0])
        
        start_idx = {}
        start_list = []
        end_idx = {}
        end_list = []
        word_dict = {}
        word_list = []
        split_sentence = sentence.split()
        for word in split_sentence:
            
            pos_word = self.pos_tagger.pos(word)            
            # pos_word = word
            for split_word in pos_word:             
                if split_word[1] in ['Noun']:
                # if split_word[1] in ['Noun']:
                    try: 
                        word_vec = self.model.wv.get_vector(split_word[0])
                        # word_vec = self.model.wv.get_vector(pos_word)
                        # word_vec = self.model.wv.get_vector(pos_word)
                    except:
                        continue
                    if cosine_similarity(keyword_vec, word_vec) > self.threshold and pos_word not in word_list:
                        results = re.finditer(word, sentence)
                        for matched_key in results:
                            # start_idx["start_dix"] = matched_key.start()
                            # end_idx["end_idx"] = matched_key.end()
                            # word_dict["word"] = word
                            start_list.append(matched_key.start())
                            end_list.append(matched_key.end())
                            word_list.append(pos_word[0][0])
        
        start_idx["start_idx"] = start_list
        end_idx["end_idx"] = end_list
        word_dict["word"] = word_list
            
        return start_idx, end_idx, word_dict

def main(keyword_list, sentence_list, model_name):
    pos_tagger = Okt()
    model = g.Doc2Vec.load(model_name)
    Fast_KS = Keyword_similarity(model, 0.35, pos_tagger)
    
    return Fast_KS.get_keyword_score(keyword_list, sentence_list)

if __name__ == '__main__':
    # 파일 불러오기
    # args = get_args()
    # sentence_data_path = args.sentence_data_path
    # sentence_file_name = '../data/' + args.sentence_file_name
    # data = pd.read_csv(sentence_file_name, encoding='utf-8')
    # sentence_list = data['sentence'].tolist()
    

    # keyword_data_path = args.keyword_data_path
    # keyword_file_name = '../data/' + args.keyword_file_name
    # data = pd.read_csv(keyword_file_name, encoding='utf-8')
    # keyword_list = data['keyword'].tolist()

    # 모델 불러오기
    
    # model_name = args.model_name
    model_name = "fast_text_ko"
    sentence_list = ["어느 곳에 빵의 값이 저렴한 지 알 수 있다.", "어느 곳에 맛이 좋은 지 알 수 있다.", "어느 곳에 가격이 낮은 지 알 수 있다."]
    keyword_list = ["가격", "품질"]
    
    main(keyword_list, sentence_list, model_name)
