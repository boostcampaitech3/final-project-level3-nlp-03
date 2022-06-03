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

def build_model(sentences, model_name="Word2Vec_Kor", size=200, context=5, min_count=5, workers=4, downsampling = 1e-3):
    
    #모델 학습
    model = word2vec.Word2Vec(sentences,
                                workers = num_workers,
                                vector_size = num_features,
                                min_count = min_word_count,
                                window = context,
                                sample = downsampling
                            )
    model.save(model_name)
    model.wv.save_word2vec_format('my.embedding', binary=False)
    return model

def load_model(model_name):
    model = word2vec.Word2Vec.load(model_name)
    return model
