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
import pickle

from soynlp.normalizer import *
from context_preprocess import *
def main():
    df1 = pd.read_csv("../data/" + "KorSTS/sts-train.tsv", sep='\t')
    df2 = pd.read_csv("../data/" + "KorSTS/sts-dev.tsv", sep='\t')
    df3 = pd.read_csv("../data/" + "KorSTS/sts-test.tsv", sep='\t')
    df4 = pd.read_csv("../data/" + "whole_answers.csv", sep=',', encoding = 'utf-8')
    data = open("../data/para_kqc_sim_data.txt", 'r', encoding='utf-8')
    lines = data.readlines()
    random.shuffle(lines)
    train_data, test_data = preprocess_data(lines)

    seed_fix(42)


    train_contexts1 = list(
        df1.iloc[i].sentence1 for i in range(len(df1))
        )  # set 은 매번 순서가 바뀌므로
    train_contexts2 = list(df1.iloc[j].sentence2 for j in range(len(df1)))
    dev_contexts1 = list(
        df2.iloc[i].sentence1 for i in range(len(df2))
            )  # set 은 매번 순서가 바뀌므로
    dev_contexts2 = list(df2.iloc[j].sentence2 for j in range(len(df2)))
    test_contexts1 = list(
        df3.iloc[i].sentence1 for i in range(len(df3))
        )
    test_contexts2 = list(df3.iloc[j].sentence2 for j in range(len(df3)))

    parpara_contexts1 = list(
        train_data.iloc[i].sent_a for i in range(len(train_data))
        )
    parpara_contexts2 = list(train_data.iloc[j].sent_b for j in range(len(train_data)))

    parpara_contexts3 = list(
        test_data.iloc[i].sent_a for i in range(len(test_data))
        )
    parpara_contexts4 = list(test_data.iloc[j].sent_b for j in range(len(test_data)))
    wai_contexts = list(
        df4["answeres"].iloc[i] for i in range(len(df4))
        )


    contexts = train_contexts1 + train_contexts2 + dev_contexts1 + dev_contexts2 + test_contexts1 + test_contexts2 + parpara_contexts1 + parpara_contexts2 + parpara_contexts3 + parpara_contexts4 + wai_contexts

    contexts = preprocess_context(contexts)
    with open("contexts.pkl", "wb") as f:
        pickle.dump(contexts, f)

if __name__ == "__main__":
    
    print("start")

    main()

    print("finish")
