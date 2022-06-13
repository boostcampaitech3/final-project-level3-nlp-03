import torch
import sys, os
import random
import numpy as np
import pandas as pd
import yaml

from transformers import Trainer, TrainingArguments, AutoConfig, AutoModelForTokenClassification
from tokenizer import get_tokenizer

from utils import seed_fix, aggregate_args_config, compute_metrics_bin, compute_metrics_reg,compute_metrics_cls
from dataset import MultiSentDataset
from arguments import get_args
from models import get_model, get_trained_model, get_trained_local_model, get_init_model
# from preprocessing import preprocess_data, tokenizing_data, get_label
from preprocessing import *
from datasets import load_dataset

import wandb

def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_fix(config['ETC']['seed'])
    wandb.login()
    exp_full_name = f"{config['MODEL']['model_name']}_{config['TRAIN_DATA']['data_name']}_{config['TASK']['type']}_{config['TRAIN']['lr']}"
    wandb.init(project='final-projects',
               name=exp_full_name)  # entity nono

    ####### DATA PROCESSING #############
    # Data type 따라서
    # data = open(config['TRAIN_DATA']['data_path'], 'r', encoding='utf-8')
    # lines = data.readlines()
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    train_data = {'sent_a':[], 'sent_b':[], 'labels':[]}
    print('yoyo')
    with open('../Dataset/snli_1.0_train.ko.tsv', "rt", encoding="utf-8") as fIn:
        lines = fIn.readlines()
        for line in lines:
            s1, s2, label = line.split('\t')
            label = label2int[label.strip()]
            train_data['sent_a'].append(s1)
            train_data['sent_b'].append(s2)
            train_data['labels'].append(label)
            # train_samples.append(InputExample(texts=[s1, s2], label=label))
    train_label = train_data['labels']

    test_data = {'sent_a':[], 'sent_b':[], 'labels':[]}
    print('yoyo')
    with open('../Dataset/snli_1.0_train.ko.tsv', "rt", encoding="utf-8") as fIn:
        lines = fIn.readlines()
        for idx, line in enumerate(lines):
            s1, s2, label = line.split('\t')
            label = label2int[label.strip()]
            test_data['sent_a'].append(s1)
            test_data['sent_b'].append(s2)
            test_data['labels'].append(label)
            if idx == 10:
                break
            # train_samples.append(InputExample(texts=[s1, s2], label=label))
    test_label = test_data['labels']

    tokenizer = get_tokenizer(config)
    tokenized_train_sentences = tokenizing_data(train_data,
                                                tokenizer,
                                                data_type=config['TRAIN_DATA']['data_name'],
                                                truncation=config['TOKENIZER']['truncation'],  # default True
                                                max_length=512)  # default 64

    tokenized_test_sentences = tokenizing_data(test_data,
                                               tokenizer,
                                               data_type=config['TRAIN_DATA']['data_name'],
                                               truncation=config['TOKENIZER']['truncation'],  # default True
                                               max_length=config['TOKENIZER']['max_length'])  # default 64

    train_dataset = MultiSentDataset(tokenized_train_sentences, train_label,
                                     data_type=config['TRAIN_DATA']['data_name'])
    test_dataset = MultiSentDataset(tokenized_test_sentences, test_label, data_type=config['TRAIN_DATA']['data_name'], )

    final_output_dir = os.path.join('klue-bert-base-korSTS_nli')
    os.makedirs(final_output_dir, exist_ok=True)

    #       lr_scheduler_type = train_args.scheduler, # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
    #         warmup_steps=train_args.warmup_steps,
    ####### metric #############

    compute_metric = compute_metrics_cls


    #######  ARGUMENTS  #############
    training_args = TrainingArguments(
        output_dir=final_output_dir,  # output directory
        num_train_epochs=config['TRAIN']['num_train_epochs'],  # total number of training epochs
        per_device_train_batch_size=config['TRAIN']['train_bs'],  # batch size per device during training
        per_device_eval_batch_size=config['TRAIN']['eval_bs'],  # batch size for evaluation
        logging_dir=config['LOGGING']['logging_dir'],  # directory for storing logs
        logging_steps=config['LOGGING']['logging_steps'],
        save_total_limit=config['LOGGING']['save_total_limit'],
        save_steps=config['TRAIN']['save_steps'],
        eval_steps=config['TRAIN']['eval_steps'],
        evaluation_strategy=config['TRAIN']['evaluation_strategy'],
        lr_scheduler_type='linear',
        # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
        # warmup_steps=train_args.warmup_steps,

    )

    #######  models  #############
    model = get_init_model('klue/bert-base', task_type='nli')


    # model.parameters
    print(model)
    model.to(device)
    #######  training  #############
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metric
    )

    trainer.train()
    # trainer.evaluate(eval_dataset=test_dataset)
    trainer.save_model(final_output_dir)
    # trainer.save_state(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    ## set config
    ## config는 향후 통합 상황을 생각해서 basic으로 넣어두었습니다.
    ## 실험의 용의성을 위해 argparse 사용도 넣어두었습니다.
    ## config 구조는 {"대문자": { "소문자" : 3, ... } 와 같습니다.
    ## config에 추가되지 않은 argparse 값은 "대문자"에 속하지 않고 그대로 들어갑니다.
    ## e.g, args.test -> config['test']

    ## config에 추가되어있는 argparse의 값을 변경하려면 config의 소문자값과 일치시켜주세요!
    LOAD_FROM = 'klue/bert-base'  # '/opt/ml/projects/final-project-level3-nlp-03/finetunning/results/klueSTS_first/checkpoint-1000'

    args = get_args()
    config_path = args.config_path  # './configs/base_config.yaml'

    ## argparse로 세팅한 값을 config 파일에 업데이트하게 됩니다.
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file)

    agg_config = aggregate_args_config(args, config)

    main(agg_config)
