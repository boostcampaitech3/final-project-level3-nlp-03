# 필요한 객체 import
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
    def tokenize(self, doc):
    # norm, stem은 optional
        if type(doc) is not str:
            return []
        return ['/'.join(t) for t in self.pos_tagger.pos(doc, norm=True, stem=True)]
    
    def get_keyword_score(self, keyword_list, sentence_list):
        results = self.keywords_sentences(keyword_list, sentence_list)
        
        keyword_num = len(results.keys())
        sentence_num = len(results.values())
        
        keyword_score = []
        for keyword, students in results.items():
            for idx, sentence in students.items():    
                keyword_score.append(len(sentence[0]))
        return results, keyword_score
        

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
        pos_keyword = pos_tagger.pos(keyword)
        keyword_vec = self.model.wv.get_vector(pos_keyword[0][0])
        
        start_idx = []
        end_idx = []
        word_list = []
        split_sentence = sentence.split()
        for word in split_sentence:
            
            pos_word = pos_tagger.pos(word)            
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
                            start_idx.append(matched_key.start())
                            end_idx.append(matched_key.end())
                        
                        word_list.append(pos_word[0][0])
            
        return start_idx, end_idx, word_list

def main(keyword_list, sentence_list, KS):
    
    return KS.keyword_score(keyword_list, sentence_list)

if __name__ == '__main__':
    # 파일 불러오기
    args = get_args()
    sentence_data_path = args.sentence_data_path
    sentence_file_name = '../data/' + args.sentence_file_name
    data = pd.read_csv(sentence_file_name, encoding='utf-8')
    sentence_list = data['sentence'].tolist()
    

    keyword_data_path = args.keyword_data_path
    keyword_file_name = '../data/' + args.keyword_file_name
    data = pd.read_csv(keyword_file_name, encoding='utf-8')
    keyword_list = data['keyword'].tolist()

    # 모델 불러오기
    pos_tagger = Okt()
    model_name = args.model_name
    model = g.Doc2Vec.load(model_name)
    Fast_KS = Keyword_similarity(model, 0.35, pos_tagger)
    main(sentence_list, keyword_list, Fast_KS)
