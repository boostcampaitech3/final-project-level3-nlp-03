# 필요한 객체 import
from ast import keyword
import pandas as pd
from konlpy.tag import Okt
import numpy as np
import re

import gensim 
import gensim.models as g

import re


def cosine_similarity(a, b):

    #return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


class Keyword_similarity:
    
    def __init__(self, model, threshold = 0.5, pos_tagger=None, lemmatizer=None):
        self.model = model
        self.threshold = threshold
        self.pos_tagger = pos_tagger
        self.lemmatizer = lemmatizer

    
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
            for split_word in pos_word:             
                if split_word[1] in ['Noun']:
                    try: 
                        word_vec = self.model.wv.get_vector(split_word[0])
                    except:
                        continue
                    if cosine_similarity(keyword_vec, word_vec) > self.threshold and pos_word not in word_list:
                        results = re.finditer(word, sentence)
                        for matched_key in results:
                            start_list.append(matched_key.start())
                            end_list.append(matched_key.end())
                            word_list.append(pos_word[0][0])
        
        start_idx["start_idx"] = start_list
        end_idx["end_idx"] = end_list
        word_dict["word"] = word_list
            
        return start_idx, end_idx, word_dict

def make_keyword_list(fastks, keyword_list, sentence_list):
    # model_name = "fast_text_ko"
    # pos_tagger = Okt()
    # model = g.Doc2Vec.load(model_name)
    # Fast_KS = Keyword_similarity(model, 0.35, pos_tagger)
    
    return fastks.get_keyword_score(keyword_list, sentence_list)
