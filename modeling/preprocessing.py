import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Union, TypeVar
# https://stackoverflow.com/questions/43890844/pythonic-type-hints-with-pandas
DataFrameStr = TypeVar("pandas.core.frame.DataFrame(str)")

def preprocess_data(lines:Union[str]) -> DataFrameStr:
    train = {'sent_a': [], 'sent_b': [], 'label': []}
    test = {'sent_a': [], 'sent_b': [], 'label': []}

    for i, line in tqdm(enumerate(lines)):
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

    train_data = pd.DataFrame({"sent_a": train['sent_a'], "sent_b": train['sent_b'], "label": train['label']})
    test_data = pd.DataFrame({"sent_a": test['sent_a'], "sent_b": test['sent_b'], "label": test['label']})

    # 중복 데이터 제거

    train_data.drop_duplicates(subset=['sent_a', 'sent_b'], inplace=True)
    test_data.drop_duplicates(subset=['sent_a', 'sent_b'], inplace=True)

    # 데이터셋 갯수 확인
    print('중복 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
    print('중복 제거 후 테스트 데이터셋 : {}'.format(len(test_data)))
    # null 데이터 제거
    train_data.replace('', np.nan, inplace=True)
    test_data.replace('', np.nan, inplace=True)

    train_data = train_data.dropna(how='any')
    test_data = test_data.dropna(how='any')

    print('null 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
    print('null 제거 후 테스트 데이터셋 : {}'.format(len(test_data)))


    return train_data, test_data

def get_label(df):
    return df['label'].values[0:]

def tokenizing_data(data, tokenizer,
                    truncation=True,
                    max_length=64):

    tokenized_sentences = tokenizer(
        list(data['sent_a'][0:]),
        list(data['sent_b'][0:]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=truncation,
        max_length=max_length
    )
    return tokenized_sentences