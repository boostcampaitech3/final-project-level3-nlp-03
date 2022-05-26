import torch
import sys
import random
import numpy as np
import pandas as pd
import yaml
from metrics import compute_metrics

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, Trainer, TrainingArguments, AutoConfig

from utils import seed_fix, aggregate_args_config
from dataset import MultiSentDataset_STS
from arguments import get_args
from models import get_model
from preprocessing import preprocess_data, tokenizing_data, get_label

def main(config):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    column_names = ["genre", "filename", "year", "id", "score", "sentence1", "sentence2"]
    df = pd.read_csv("/opt/ml/KorNLUDatasets/KorSTS/sts-train.tsv", header=None, delimiter="\t", names=column_names)
    
    train = {'sent_a':[], 'sent_b':[], 'label':[]}
    test = {'sent_a':[], 'sent_b':[], 'label':[]}

    for i in range(1, len(df)):
        train['sent_a'].append(df['sentence1'][i])
        train['sent_b'].append(df['sentence2'][i])
        train['label'].append(float(df['score'][i]))
    
    test_df = pd.read_csv("/opt/ml/KorNLUDatasets/KorSTS/sts-dev.tsv", header=None, delimiter="\t", names=column_names)
    
    for i in range(1, len(test_df)):
        test['sent_a'].append(test_df['sentence1'][i])
        test['sent_b'].append(test_df['sentence2'][i])
        test['label'].append(float(test_df['score'][i]))
        
    train_data = pd.DataFrame({"sent_a":train['sent_a'], "sent_b":train['sent_b'], "label":train['label']})
    test_data = pd.DataFrame({"sent_a":test['sent_a'], "sent_b":test['sent_b'], "label":test['label']})

    # 중복 데이터 제거
    
    train_data.drop_duplicates(subset=['sent_a', 'sent_b'], inplace= True)
    test_data.drop_duplicates(subset=['sent_a', 'sent_b'], inplace= True)

    train_data.replace('', np.nan, inplace=True)
    test_data.replace('', np.nan, inplace=True)

    train_data = train_data.dropna(how = 'any')
    test_data = test_data.dropna(how = 'any')

    # Store the model we want to use

    MODEL_NAME = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_train_sentences = tokenizer(
        list(train_data['sent_a'][0:]),
        list(train_data['sent_b'][0:]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=128
        )
    
    tokenized_test_sentences = tokenizer(
        list(test_data['sent_a'][0:]),
        list(test_data['sent_b'][0:]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=128
        )
    
    train_label = train_data['label'].values[0:]
    test_label = test_data['label'].values[0:]

        
    train_dataset = MultiSentDataset_STS(tokenized_train_sentences, train_label)
    test_dataset = MultiSentDataset_STS(tokenized_test_sentences, test_label)

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
    model.to(device)
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,             # evaluation dataset
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
    config_path = './configs/sts_config.yaml'

    ## argparse로 세팅한 값을 config 파일에 업데이트하게 됩니다.
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file)

    agg_config = aggregate_args_config(args, config)

    main(agg_config)
