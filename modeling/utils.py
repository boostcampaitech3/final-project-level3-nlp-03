import torch
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import copy

def seed_fix(seed):
    """seed setting 함수"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def aggregate_args_config(args, config):
    args_dict= vars(args)
    first_capital_keys = list(config.keys()) # list로 안하면 추가되는 key가 계속 반영됨

    for arg_key, arg_val in args_dict.items():
        check = False

        for capital in first_capital_keys:
            if arg_key in config[capital]:
                config[capital][arg_key] = arg_val
                check= True
                break
        if not check:
            config.update({arg_key: arg_val})
    return config

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


