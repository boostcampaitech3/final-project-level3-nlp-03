import torch
import sys
import random
import numpy as np
import pandas as pd
import yaml

from transformers import Trainer,TrainingArguments, AutoConfig
from tokenizer import get_tokenizer

from dataset import MultiSentDataset
from arguments import get_args
from models import get_model
from preprocessing import preprocess_data, tokenizing_data, get_label
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer, BertTokenizer

def upload_model(config):

    HUGGINGFACE_AUTH_TOKEN = '' # https://huggingface.co/settings/token
    MODEL_SAVE_REPO = 'reg_trained'  # ex) 'my-bert-fine-tuned'
    MODEL_SAVED_PATH =  '/opt/ml/projects/final-project-level3-nlp-03/modeling/results/checkpoint-1000' # "klue/bert-base"#"/opt/ml/projects/final-project-level3-nlp-03/modeling/results/checkpoint-1000"

    # 학습완료된 모델과 토크나이저 파일 로드
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVED_PATH)
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

def upload_sentence_transformer(config):
    HUGGINGFACE_AUTH_TOKEN = '' # https://huggingface.co/settings/token
    MODEL_SAVE_REPO = 'sbert-kornli-knoSTS-trained'  # ex) 'my-bert-fine-tuned'
    MODEL_SAVED_PATH =  '/opt/ml/projects/final-project-level3-nlp-03/finetunning/output/klue-bert-nli_sts_cosine' # "klue/bert-base"#"/opt/ml/projects/final-project-level3-nlp-03/modeling/results/checkpoint-1000"

    # 학습완료된 모델과 토크나이저 파일 로드
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_SAVED_PATH)
    model.save_to_hub(repo_name=MODEL_SAVE_REPO)

def check_sentence_transforemrs(config):
    from sentence_transformers import SentenceTransformer
    MODEL_SAVE_REPO = 'kimcando/sbert-kornli-knoSTS-trained'
    model = SentenceTransformer(MODEL_SAVE_REPO)
    print('done!')

def check(config):


    HUGGINGFACE_AUTH_TOKEN = '' # https://huggingface.co/settings/token
    LOAD_FROM = 'kimcando/sbert-kornli+knoSTS-trained'

    #######  load from huggingface  #############
    # 만약 private repo면 use_auth_token 사용 필요
    model = AutoModelForSequenceClassification.from_pretrained(
        LOAD_FROM)
    breakpoint()
    # tokenizer = AutoTokenizer.from_pretrained(LOAD_FROM)
    print('downloaded from your repo!')


if __name__ == "__main__":
    agg_config = None

    # upload_model(agg_config)
    # check(agg_config)
    # upload_sentence_transformer(agg_config)
    check_sentence_transforemrs(agg_config)