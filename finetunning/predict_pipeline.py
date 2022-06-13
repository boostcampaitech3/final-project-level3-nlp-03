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
import torch.nn.functional as F


LABEL_MAP = {
    'LABEL_0': '0', # disimilar
    'LABEL_1' : '1', # similar
}


def pipeline_test_bin():
    # 속도 최적화 고민이 필요할 것 같습니다.
    pairs = pd.read_csv(VALID_DATA_PATH).drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    pipe = pipeline('text-classification',
                    model= LOAD_FROM,
                    device = 0) # CUDA 숫자로 넣어줘야함

    # TODO 1(다음주)
    # 앗!! pipeline이 text pair로는 제대로 지원 안해주는 것 같네요..!
    # 임시방편으로 해당 링크 코드 참고합니다.
    # https://pythontechworld.com/issue/huggingface/transformers/17305

    data = {'sent_a':[], 'sent_b':[], 'labels':[], 'pred_labels':[],'scores':[]}

    # process in SINGLE PAIR -> TODO 2 속도 성능 향상 어떻게 하면 좋을지 고민 필요(다음주)

    for student_id, (gold_answer, answer,label) in pairs.iterrows():
        inputs = [gold_answer, answer]
        data['sent_a'].append(gold_answer)
        data['sent_b'].append(answer)

        tokenized_inputs = pipe.preprocess([inputs])
        res = pipe.forward(tokenized_inputs)
        out = pipe.postprocess(res)
        #print(out) # {'label': 'LABEL_1', 'score': 0.5062992572784424}

        sim_score = out['score']

        data['scores'].append(sim_score)
        data['labels'].append(label)
        data['pred_labels'].append(int(LABEL_MAP[out['label']]))

    return_df = pd.DataFrame(data)
    return_df.to_csv(os.path.join(SAVE_BASE_PATH,'korSTS.csv'))

def model_test_bin(csv_save_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(LOAD_FROM, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZED_FROM)

    model.eval()
    try:
        pairs = pd.read_csv(VALID_DATA_PATH).drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    except:
        pairs = pd.read_csv(VALID_DATA_PATH)


    sent_a = pairs['sent_a'].tolist()
    sent_b = pairs['sent_b'].tolist()
    try:
        org_labels = pairs['labels'].tolist()
    except:
        org_labels = pairs['Labels'].tolist()

    tokenized_sent = tokenizer(
            sent_a,
            sent_b,
            padding = True,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=64
    )
    tokenized_sent.to('cuda:0')
    with torch.no_grad():# 그라디엔트 계산 비활성화
        outputs = model(
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )
        # 0.2 0.8 -> 1 -> (지금) 0.8
        # 0.1 0.9 - >1 -> (지금) 0.9
        # 0.7 0.3 -> 0 -> (지금) 0.3
        logits = outputs[0]
        probs = F.softmax(logits, dim=1).cpu()
        labels = torch.argmax(probs, dim=1)
        labels = labels.detach().cpu().numpy().tolist()
        # pred_scores = probs[torch.arange(len(labels)), labels] -> 0,1 각각에 대응되는 값들 return 해준다
        pred_scores = probs[torch.arange(len(labels)), [1]*len(labels)]
    data = {'sent_a':sent_a, 'sent_b':sent_b, 'labels':org_labels,'pred_labels':labels, 'scores':pred_scores}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(SAVE_BASE_PATH,csv_save_name))


def model_test_reg(csv_save_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(LOAD_FROM,num_labels=1).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZED_FROM)

    model.eval()
    try:
        pairs = pd.read_csv(VALID_DATA_PATH).drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    except:
        pairs = pd.read_csv(VALID_DATA_PATH)
    sent_a = pairs['sent_a'].tolist()
    sent_b = pairs['sent_b'].tolist()
    org_labels = pairs['labels'].tolist()

    tokenized_sent = tokenizer(
        sent_a,
        sent_b,
        padding=True,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        max_length=64
    )
    tokenized_sent.to('cuda:0')
    with torch.no_grad():  # 그라디엔트 계산 비활성화
        outputs = model(
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
        )

        logits = outputs[0]
        normalized_scores = (logits/5).cpu().squeeze().numpy()
        org_scores = logits.cpu().squeeze().numpy()
    data = {'sent_a': sent_a, 'sent_b': sent_b, 'labels': org_labels, 'normalized_scores': normalized_scores.tolist(), 'org_scores': org_scores.tolist()}
    threshold = [0.6,0.7,0.8,0.9]
    for thr in threshold:
        new = np.where(org_scores>thr, 1,0).tolist()
        # new = np.where(normalized_scores > thr, 1, 0).tolist()
        data.update({f'thr_{thr}':new})

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(SAVE_BASE_PATH,csv_save_name))

def check_bin_acc(df):
    threshold = [0.6, 0.7, 0.8, 0.9]
    for thr in threshold:
        cnt = 0

        for idx in range(len(df)):
            val = df['scores'][idx]
            if val >= thr:
                label = 1
            else:
                label = 0

            if df['labels'][idx] == label:
                cnt += 1
        print(f'{thr} : {cnt / len(df)}')

def check_reg_acc(df):
    threshold = [0.6, 0.7, 0.8, 0.9]
    for thr in threshold:
        cnt = 0

        for idx in range(len(df)):
            # normalized_scores
            val = df['org_scores'][idx]
            if val >= thr:
                label = 1
            else:
                label = 0

            if df['labels'][idx] == label:
                cnt += 1

        print(f'{thr} : {cnt / len(df)}')

def model_test_reg_korsts(csv_save_name):
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(LOAD_FROM,num_labels=1).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZED_FROM)

    model.eval()
    try:
        pairs = pd.read_csv(VALID_DATA_PATH).drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    except:
        pairs = pd.read_csv(VALID_DATA_PATH)
    sent_a = pairs['sent_a'].tolist()
    sent_b = pairs['sent_b'].tolist()
    org_labels = pairs['labels'].tolist()

    tokenized_sent = tokenizer(
        sent_a,
        sent_b,
        padding=True,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        max_length=64
    )
    tokenized_sent.to('cuda:0')
    with torch.no_grad():  # 그라디엔트 계산 비활성화
        outputs = model(
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
        )

        logits = outputs[0]

        normalized_scores = (logits/5).cpu().squeeze().numpy()
        org_scores = F.sigmoid(logits).cpu().squeeze().numpy()
        # org_scores = logits.cpu().squeeze().numpy()
      
    data = {'sent_a': sent_a, 'sent_b': sent_b, 'labels': org_labels, 'normalized_scores': normalized_scores.tolist(), 'org_scores': org_scores.tolist()}
    # threshold = [0.6,0.7,0.8,0.9]
    # for thr in threshold:
    #     new = np.where(normalized_scores>thr, 1,0).tolist()
    #     data.update({f'thr_{thr}':new})

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(SAVE_BASE_PATH,csv_save_name))

if __name__=='__main__':

    # pipeline_test_bin()
    VALID_DATA_PATH = '/opt/ml/projects/tunning_data/demo_eval.csv'
    # VALID_DATA_PATH = '/opt/ml/projects/tunning_data/validation_v1.csv'
    SAVE_BASE_PATH = '/opt/ml/projects/final-project-level3-nlp-03/finetunning/inference_results'
    os.makedirs(SAVE_BASE_PATH,exist_ok=True)
    TOKENIZED_FROM = 'klue/bert-base'
    LOAD_FROM = '/opt/ml/projects/final-project-level3-nlp-03/finetunning/results/korsts_first'
    csv_save_name = 'klue-bert-korSTS-first-demo_eval.csv'

    model_test_reg(csv_save_name) # model_test_reg_korsts(csv_save_name)
    # model_test_reg_korsts(csv_save_name)
    FILE_NAME =  csv_save_name # 'klueSTS_reg_1000_gen_bin_final.csv' # 'klueSTS_reg_1000_para_bin_final.csv'
    df = pd.read_csv(os.path.join(SAVE_BASE_PATH, FILE_NAME))
    check_reg_acc(df)

    # model_test_bin(csv_save_name)
    # FILE_NAME = csv_save_name  # 'klueSTS_reg_1000_gen_bin_final.csv' # 'klueSTS_reg_1000_para_bin_final.csv'
    # df = pd.read_csv(os.path.join(SAVE_BASE_PATH, FILE_NAME))
    # check_bin_acc(df)










