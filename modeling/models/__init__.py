from transformers import AutoModel, AutoModelForSequenceClassification

def get_model(config):
    model = AutoModelForSequenceClassification.from_pretrained(
        config['MODEL']['model_name'],
        num_labels = config['MODEL']['num_labels'],
        ignore_mismatched_sizes = True        
        )
        
    return model