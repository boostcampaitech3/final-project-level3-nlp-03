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
  
  PATH = "/opt/ml/results2/checkpoint-1500/"
  model = load_model("klue/roberta-large")
  model.load_state_dict(torch.load(PATH + "pytorch_model.bin"))  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
  
  tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
  sentA = pd.read_csv("../data/example.csv")["answer"].to_list()[:10]
  sentB = ["더 질좋은 제품을 생산할 수 있으며 더 좋은 제품을 소비자에게 제공하려 힘쓸것이다."]*len(sentA)  
  result, logits = sentences_predict(model, tokenizer, sentA, sentB)
  softmax = torch.nn.Softmax(dim=1)
  prob = softmax(torch.tensor(logits))
  print(logits)
  print(prob)

if __name__ == "__main__":
    main()