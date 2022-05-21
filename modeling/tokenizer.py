from transformers import AutoTokenizer, BertTokenizer

def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config['MODEL']['model_name'])
    return tokenizer