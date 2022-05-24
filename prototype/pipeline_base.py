import sys,os
import random
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# DL
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, Trainer, TrainingArguments, AutoConfig

from transformers import pipeline
from datasets import load_dataset
# import streamlit as st
# from typing import Tuple

LABEL_MAP = {
    'LABEL0': 'similar',
    'LABEL1' : 'disimmilar',
}

LOAD_FROM = 'kimcando/final_projects'

def checker(sent_a, key_words=None):
    # 디테일한 사항은 구현 필요
    # 본 함수는 테스트용
    total_key_words = 3
    correct = 1
    matching_key_info= {
        'keword': ['aa', 'bb','cc'],
        'start_idx' : [3, 6, 10],
        'end_idx' : [4, 7, 15]
    }
    return round(correct/total_key_words, 2) , matching_key_info


def main(config, pairs):
    # 속도 최적화 고민이 필요할 것 같습니다.

    pipe = pipeline('text-classification',
                    model= LOAD_FROM,
                    device = 0) # CUDA 숫자로 넣어줘야함

    # TODO 1(다음주)
    # 앗!! pipeline이 text pair로는 제대로 지원 안해주는 것 같네요..!
    # 임시방편으로 해당 링크 코드 참고합니다.
    # https://pythontechworld.com/issue/huggingface/transformers/17305

    meta_info = {'subject':'과학', 'problem_idx':0, 'keyword':[], 'questions':[], 'gold_answer': ''}
    return_dict = {'student_id':[], 'answers':[],
                    'keyword_score':[], 'sim_score':[], 'total_score':[]}

    # studnet_id 한번 더 넣은거는 혹시 몰라서..! 편한대로 변경하면 됨
    # 혹시 성능 향상을 위해 multiprocessing 모듈로 처리하게 되면
    # return_dict와 ui_dict를 concat할 때 student_id값을 key로 union하면 좋을 것 같아서
    ui_dict = {'student_id':[], 'key_info':[]}

    # process in SINGLE PAIR -> TODO 2 속도 성능 향상 어떻게 하면 좋을지 고민 필요(다음주)
    for student_id, (gold_answer, answer) in pairs.iterrows():
        inputs = [gold_answer, answer]
        breakpoint()
        tokenized_inputs = pipe.preprocess([inputs])
        res = pipe.forward(tokenized_inputs)
        out = pipe.postprocess(res)
        #print(out) # {'label': 'LABEL_1', 'score': 0.5062992572784424}
        sim_score = out['score']

        check_score, match_info = checker(answer) # TODO

        return_dict['student_id'].append(student_id)
        return_dict['answers'].append(answers)
        return_dict['sim_score'].append(sim_score)
        return_dict['check_score'].append(check_score)

        ui_dict['student_id'].append(student_id)
        # 학생 별로 각각 keyword, start_idx, end_idx 리스트를 가짐.
        # e.g, keyword[idx]의 keyword의 시작점은 start_idx[idx], 끝점은 end_idx[idx]
        ui_dict['key_info'].append(match_info)

    # json으로 할 수도 있고
    return_df = pd.DataFrame(return_dict)
    return_df.to_csv('for_return.csv')

    ui_df = pd.concat([return_df, pd.DataFrame(ui_dict) ], axis=1)
    ui_df.to_csv('for_ui.csv')

    # TODO3(다음주)
    # process in BATCH MODE -> padding, truncation 길이 확인 필요!

if __name__=='__main__':
    # 환경변수 설정
    cur_pwd = os.getcwd()
    module_upper_path = cur_pwd[:-len('/prototype')]
    sys.path.append(module_upper_path)
    from modeling.models import get_model
    from modeling.arguments import get_args
    from modeling.dataset import MultiSentDataset
    from modeling.utils import aggregate_args_config,compute_metrics

    args = get_args()
    config_path = '../modeling/configs/base_config.yaml'
    ## argparse로 세팅한 값을 config 파일에 업데이트하게 됩니다.
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file)

    # 만약 텍스트 파일로 테스트 하고 싶다면
    text = pd.read_csv('/opt/ml/projects/final-project-level3-nlp-03/data/dummy_data/pipeline_test.txt',
                       sep=',',
                       header=None)

    agg_config = aggregate_args_config(args, config)
    main(agg_config, text)











