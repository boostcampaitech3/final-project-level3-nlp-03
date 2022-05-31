import torch
import sys
import random
import numpy as np
import pandas as pd

import streamlit as st
import yaml
from typing import Tuple

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, Trainer, TrainingArguments, AutoConfig


def load_model(MODEL_NAME) -> AutoModelForSequenceClassification:
    # with open("config.yaml") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    
    return model

def sentences_predict(model, tokenizer, sent_A, sent_B):
    model.eval()
    tokenized_sent = tokenizer(
            sent_A,
            sent_B,
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

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits)

    if result == 0:
      result = 'non_similar'
    elif result == 1:
      result = 'similar'
    return result, logits

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # PATH = "/opt/ml/projects/final-project-level3-nlp-03/modeling/results_para_korsentence/checkpoint-4000/"  # "/opt/ml/projects/final-project-level3-nlp-03/modeling/results_para/checkpoint-1600/"
    # model = load_model("klue/roberta-large")
    # model.load_state_dict(torch.load(PATH + "pytorch_model.bin"))  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    #
    # tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    #
    # sentA = pd.read_csv("/opt/ml/projects/final-project-level3-nlp-03/data/validation.csv")["student_ans"].to_list()
    # sentB = pd.read_csv("/opt/ml/projects/final-project-level3-nlp-03/data/validation.csv")["right_ans"].to_list()
    # result, logits = sentences_predict(model, tokenizer, sentA, sentB)
    # softmax = torch.nn.Softmax(dim=1)
    # prob = softmax(torch.tensor(logits))
    # ans = prob.argmax(dim=1)
    # pd.DataFrame({"sentA": sentA, "prob": prob.detach().cpu().numpy()[:, 1], "ans": ans}).to_csv(
    #     "../data/validation_kor.csv")



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    PATH = "/opt/ml/projects/final-project-level3-nlp-03/modeling/results_para_korsentence/checkpoint-4000/" # "/opt/ml/projects/final-project-level3-nlp-03/modeling/results_para/" #"/opt/ml/projects/final-project-level3-nlp-03/modeling/results_para/checkpoint-1600/"
    model = load_model("klue/roberta-large")
    model.load_state_dict(torch.load(PATH + "pytorch_model.bin"))  # 전체 모델을 통째로 불러옴, 클래스 선언 필수

    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    sentA = pd.read_csv("/opt/ml/projects/final-project-level3-nlp-03/data/example.csv", sep='\t')["answer"].to_list()
    sentB = ["제과점이 경쟁을 하게 되면 제품의 가격은 낮아지고 품질은 좋아진다."]*len(sentA)
    result, logits = sentences_predict(model, tokenizer, sentA, sentB)
    softmax = torch.nn.Softmax(dim=1)
    prob = softmax(torch.tensor(logits))
    ans = prob.argmax(dim=1)
    pd.DataFrame({"sentA" : sentA, "prob" : prob.detach().cpu().numpy()[:,1], "ans": ans}).to_csv("../data/example_prob3.csv")

if __name__ == "__main__":
    main()