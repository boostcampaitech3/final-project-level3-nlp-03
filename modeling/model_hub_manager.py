import torch
import sys
import random
import numpy as np
import pandas as pd
import yaml

from transformers import Trainer,TrainingArguments, AutoConfig
from tokenizer import get_tokenizer

from utils import seed_fix, aggregate_args_config, compute_metrics
from dataset import MultiSentDataset
from arguments import get_args
from models import get_model
from preprocessing import preprocess_data, tokenizing_data, get_label
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer, BertTokenizer

def upload_model(config):

    HUGGINGFACE_AUTH_TOKEN = '' # https://huggingface.co/settings/token
    MODEL_SAVE_REPO = 'para_test_4800'  # ex) 'my-bert-fine-tuned'
    MODEL_SAVED_PATH = "/opt/ml/projects/final-project-level3-nlp-03/modeling/results_para_korsentence/checkpoint-4800"#"/opt/ml/projects/final-project-level3-nlp-03/modeling/results/checkpoint-1000"

    # 학습완료된 모델과 토크나이저 파일 로드
    # tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SAVED_PATH)

    ## Push to huggingface-hub : public일 때
    model.push_to_hub(
        MODEL_SAVE_REPO,
        use_temp_dir=True,
        # use_auth_token=HUGGINGFACE_AUTH_TOKEN
    )
    # tokenizer.push_to_hub(
    #     MODEL_SAVE_REPO,
    #     use_temp_dir=True,
    #     # use_auth_token=HUGGINGFACE_AUTH_TOKEN
    # )
    # Private일 때
    # model.push_to_hub(
    #     MODEL_SAVE_REPO,
    #     use_temp_dir=True,
    #     use_auth_token=HUGGINGFACE_AUTH_TOKEN
    # )
    # tokenizer.push_to_hub(
    #     MODEL_SAVE_REPO,
    #     use_temp_dir=True,
    #     use_auth_token=HUGGINGFACE_AUTH_TOKEN
    # )

def check(config):


    HUGGINGFACE_AUTH_TOKEN = '' # https://huggingface.co/settings/token
    LOAD_FROM = 'kimcando/para_test_4800'

    #######  load from huggingface  #############
    # 만약 private repo면 use_auth_token 사용 필요
    model = AutoModelForSequenceClassification.from_pretrained(
        LOAD_FROM)
    tokenizer = AutoTokenizer.from_pretrained(LOAD_FROM)
    print('downloaded from your repo!')


if __name__ == "__main__":
    args = get_args()
    config_path = './configs/base_config.yaml'

    ## argparse로 세팅한 값을 config 파일에 업데이트하게 됩니다.
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file)

    agg_config = aggregate_args_config(args, config)
    check(agg_config)
    # upload_model(agg_config)
