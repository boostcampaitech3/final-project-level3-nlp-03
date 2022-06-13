import torch
import numpy as np
import random
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score
from scipy import stats
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

def compute_metrics_bin(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    pred_scores = pred.predictions[torch.arange(len(preds)), preds]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    pearson_coeff = stats.pearsonr(labels, pred_scores)
    spear_coeff = stats.spearmanr(labels, pred_scores)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'pearson' : pearson_coeff[0],
        'spear': spear_coeff[0]
    }

def compute_metrics_cls(pred):
    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_metrics_reg(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions, squared=False)
    pearson_coeff = stats.pearsonr(labels, predictions.squeeze())
    spear_coeff = stats.spearmanr(labels, predictions.squeeze())

    return {"mse": mse,
            'pearson': pearson_coeff[0],
            'spear': spear_coeff[0]
            }

