from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='klue/bert-base')
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--evaluation_steps', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=4)
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# model_name = './output/training_nli_'+args.model.replace("/", "-")

train_batch_size = args.batch
num_epochs = args.epochs

model_save_path = 'output/klue-bert-nli-klueSTS' #'output/para+MRC-bin-ko'
model_name = '/opt/ml/projects/final-project-level3-nlp-03/finetunning/output/training_nli_klue-bert-base' # '/opt/ml/projects/final-project-level3-nlp-03/finetunning/output/kosbert-klue-bert-base_cosine' # 'sentence-transformers/xlm-r-large-en-ko-nli-ststb' # ststb는 뭐임?
model = SentenceTransformer(model_name)

logging.info("Read STSbenchmark train dataset")

train_samples = []
valid_samples = []
test_samples = []
train_path = '/opt/ml/projects/tunning_data/klueSTS_train.csv'
# train_data = pd.read_csv(train_path).drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
train_data = pd.read_csv(train_path)

for data_idx, data in train_data.iterrows():
    score = float(data.labels)
    # score = float(data.real_score)
    s1 = data.sent_a
    s2 = data.sent_b
    train_samples.append(InputExample(texts= [s1,s2], label=score))

# valid_path = '/opt/ml/projects/tunning_data/validation_v1.csv' # '/opt/ml/projects/tunning_data/klueSTS_valid.csv'
# # valid_data = pd.read_csv(valid_path).drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
# valid_data = pd.read_csv(valid_path)
# for data_idx, data in valid_data.iterrows():
#     score = float(data.labels)
#     # score = float(data.real_score)
#     s1 = data.sent_a
#     s2 = data.sent_b
#     valid_samples.append(InputExample(texts= [s1,s2], label=score))
#
# test_path = '/opt/ml/projects/tunning_data/validation_v1.csv'
# # test_data = pd.read_csv(test_path).drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
# test_data = pd.read_csv(test_path)
# for data_idx, data in test_data.iterrows():
#     score = float(data.labels)
#     s1 = data.sent_a
#     s2 = data.sent_b
#     test_samples.append(InputExample(texts= [s1,s2], label=score))

with open('../Dataset/tune_sts_dev.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        valid_samples.append(InputExample(texts= [s1,s2], label=score))

with open('../Dataset/tune_sts_test.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        test_samples.append(InputExample(texts= [s1,s2], label=score))

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


# Development set: Measure correlation between cosine score and gold labels
logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(valid_samples, name='sts-dev')

warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=args.evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)