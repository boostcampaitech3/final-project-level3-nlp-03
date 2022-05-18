import torch
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, Trainer, TrainingArguments, AutoConfig

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = open('./para_kqc_sim_data.txt', 'r', encoding='utf-8')
    lines = data.readlines()
    
    random.shuffle(lines)
    train = {'sent_a':[], 'sent_b':[], 'label':[]}
    test = {'sent_a':[], 'sent_b':[], 'label':[]}
    
    for i, line in enumerate(lines):
        if i < len(lines) * 0.8:
            line = line.strip()
            train['sent_a'].append(line.split('\t')[0])
            train['sent_b'].append(line.split('\t')[1])
            train['label'].append(int(line.split('\t')[2]))
        else:
            line = line.strip()
            test['sent_a'].append(line.split('\t')[0])
            test['sent_b'].append(line.split('\t')[1])
            test['label'].append(int(line.split('\t')[2]))
    
    train_data = pd.DataFrame({"sent_a":train['sent_a'], "sent_b":train['sent_b'], "label":train['label']})
    test_data = pd.DataFrame({"sent_a":test['sent_a'], "sent_b":test['sent_b'], "label":test['label']})

    # 중복 데이터 제거
    
    train_data.drop_duplicates(subset=['sent_a', 'sent_b'], inplace= True)
    test_data.drop_duplicates(subset=['sent_a', 'sent_b'], inplace= True)

    # 데이터셋 갯수 확인
    print('중복 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
    print('중복 제거 후 테스트 데이터셋 : {}'.format(len(test_data)))
    # null 데이터 제거
    train_data.replace('', np.nan, inplace=True)
    test_data.replace('', np.nan, inplace=True)

    train_data = train_data.dropna(how = 'any')
    test_data = test_data.dropna(how = 'any')

    print('null 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
    print('null 제거 후 테스트 데이터셋 : {}'.format(len(test_data)))

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
        max_length=64
        )
    
    tokenized_test_sentences = tokenizer(
        list(test_data['sent_a'][0:]),
        list(test_data['sent_b'][0:]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=64
        )
    
    train_label = train_data['label'].values[0:]
    test_label = test_data['label'].values[0:]

    class MultiSentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
        
    train_dataset = MultiSentDataset(tokenized_train_sentences, train_label)
    test_dataset = MultiSentDataset(tokenized_test_sentences, test_label)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=6,              # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        logging_dir='./logs',            # directory for storing logs
        logging_steps=500,
        save_total_limit=2,
    )


    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) 
    model.parameters
    model.to(device)

    

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,             # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate(eval_dataset=test_dataset)
    trainer.save_model('./results')

if __name__ == "__main__":
    main()