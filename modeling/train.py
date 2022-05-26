import torch
import sys
import random
import numpy as np
import pandas as pd
import yaml

from transformers import Trainer,TrainingArguments, AutoConfig, AutoModelForTokenClassification
from tokenizer import get_tokenizer

from utils import seed_fix, aggregate_args_config, compute_metrics
from dataset import MultiSentDataset
from arguments import get_args
from models import get_model
from preprocessing import preprocess_data, tokenizing_data, get_label

def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = open(config['TRAIN_DATA']['data_path'], 'r', encoding='utf-8')
    lines = data.readlines()
    seed_fix(config['ETC']['seed'])

    # TODO : valid, train 나눈 데이터로 만들어야할 것 같아요

    ####### DATA PROCESSING #############
    random.shuffle(lines)
    train_data, test_data = preprocess_data(lines)
    tokenizer = get_tokenizer(config)
    tokenized_train_sentences = tokenizing_data(train_data,
                                                tokenizer,
                                                truncation=config['TOKENIZER']['truncation'], # default True
                                                max_length=config['TOKENIZER']['max_length'] ) # default 64

    tokenized_test_sentences = tokenizing_data(test_data,
                                                tokenizer,
                                                truncation=config['TOKENIZER']['truncation'],  # default True
                                                max_length=config['TOKENIZER']['max_length'])  # default 64


    train_label = get_label(train_data)
    test_label = get_label(test_data)

    train_dataset = MultiSentDataset(tokenized_train_sentences, train_label)
    test_dataset = MultiSentDataset(tokenized_test_sentences, test_label)

    #######  ARGUMENTS  #############
    training_args = TrainingArguments(
        output_dir=config['OUTPUT']['model_save'],  # output directory
        num_train_epochs=config['TRAIN']['num_train_epochs'],  # total number of training epochs
        per_device_train_batch_size=config['TRAIN']['train_bs'],  # batch size per device during training
        per_device_eval_batch_size=config['TRAIN']['eval_bs'],  # batch size for evaluation
        logging_dir=config['LOGGING']['logging_dir'],  # directory for storing logs
        logging_steps=config['LOGGING']['logging_steps'],
        save_total_limit=config['LOGGING']['save_total_limit'],
        save_steps=config['TRAIN']['save_steps'],
        eval_steps=config['TRAIN']['eval_steps'],
        evaluation_strategy=config['TRAIN']['evaluation_strategy'],
    )

    #######  models  #############
    model = get_model(config)
    # model_checkpoint = config['MODEL']['model_name']
    # model_config = AutoConfig.from_pretrained(model_checkpoint)
    # model_config.num_labels = 2
    # model = AutoModelForTokenClassification.from_config(model_config)
    
    # model.parameters
    model.to(device)

    #######  training  #############
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate(eval_dataset=test_dataset)
    trainer.save_model(config['OUTPUT']['model_save'])


if __name__ == "__main__":

    ## set config
    ## config는 향후 통합 상황을 생각해서 basic으로 넣어두었습니다.
    ## 실험의 용의성을 위해 argparse 사용도 넣어두었습니다.
    ## config 구조는 {"대문자": { "소문자" : 3, ... } 와 같습니다.
    ## config에 추가되지 않은 argparse 값은 "대문자"에 속하지 않고 그대로 들어갑니다.
    ## e.g, args.test -> config['test']

    ## config에 추가되어있는 argparse의 값을 변경하려면 config의 소문자값과 일치시켜주세요!

    args = get_args()
    config_path = './configs/base_config.yaml'

    ## argparse로 세팅한 값을 config 파일에 업데이트하게 됩니다.
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file)

    agg_config = aggregate_args_config(args, config)

    main(agg_config)
