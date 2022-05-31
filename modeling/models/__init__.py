from transformers import AutoModel, AutoModelForSequenceClassification
import torch.nn as nn

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

def from_to_cls(model_path, cls_out=2):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    in_feature_dim = model.classifier.out_proj.in_features
    new_proj = nn.Linear(in_feature_dim, cls_out, bias=True)
    model.classifier.out_proj = new_proj
    model.num_labels = cls_out
    model.config.update({'problem_type': 'single_label_classification'})

    return model