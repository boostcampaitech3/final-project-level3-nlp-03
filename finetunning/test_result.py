import torch
import sys, os
import random
import numpy as np
import pandas as pd
import yaml

from transformers import Trainer,TrainingArguments, AutoConfig, AutoModelForSequenceClassification
from tokenizer import get_tokenizer

from utils import seed_fix, aggregate_args_config, compute_metrics
from dataset import MultiSentDataset
from arguments import get_args
from models import get_model, get_trained_model
# from preprocessing import preprocess_data, tokenizing_data, get_label
from preprocessing import *

# https://huggingface.co/datasets/klue
model = AutoModelForSequenceClassification.from_pretrained('Huffon/klue-roberta-base-nli')
breakpoint()
print(model)