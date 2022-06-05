from transformers import AutoModel, AutoModelForSequenceClassification
import torch.nn as nn
from transformers import BertForSequenceClassification,RobertaForSequenceClassification

def get_trained_model(config, LOAD_NAME, pre_task = 'reg', this_task = 'bin', load_from_huggingface=True):
    if load_from_huggingface and this_task == 'bin' and (pre_task=='reg' or pre_task=='cls'):
        model = from_to_cls(LOAD_NAME)

    elif load_from_huggingface and this_task == 'bin' and pre_task == 'bin':
        model = AutoModelForSequenceClassification.from_pretrained(
            config['MODEL']['model_name']
        )

    else:
        # TODO
        model = AutoModelForSequenceClassification.from_pretrained(
            config['MODEL']['model_name'],
            num_labels = config['MODEL']['num_labels'],
            ignore_mismatched_sizes = True
            )
        
    return model


def get_model(config):

    model = AutoModelForSequenceClassification.from_pretrained(
            config['MODEL']['model_name'],
            num_labels=config['MODEL']['num_labels'],
            ignore_mismatched_sizes=True)

    return model

def get_trained_local_model(model_path, cls_out):
    model = from_to_cls(model_path, cls_out)
    return model

def get_init_model(model_path, task_type):
    if task_type == 'nli':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels = 3)
    elif task_type == 'reg':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels = 1)

    elif task_type =='re':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels = 27)

    elif task_type =='mrc':
        # TODO
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels = 27)

    return model

def from_to_cls(model_path, cls_out=2):

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if isinstance(model, BertForSequenceClassification):
        in_feature_dim = model.classifier.in_features
        new_proj = nn.Linear(in_feature_dim, cls_out, bias=True)
        model.classifier = new_proj

    elif isinstance(mode, RobertaForSequenceClassification):
        in_feature_dim = model.classifier.out_proj.in_features
        new_proj = nn.Linear(in_feature_dim, cls_out, bias=True)
        model.classifier.out_proj = new_proj

    else:
        print('in the from_to_cls function! not defined class')
        raise NotImplementedError


    model.num_labels = cls_out
    model.config.update({'problem_type': 'single_label_classification'})

    return model