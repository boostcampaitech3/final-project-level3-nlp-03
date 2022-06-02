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

def preprocess_gen_data(data, train_num=12000)-> DataFrameStr:

    train_num = int(len(data)*0.8)
    train_data = data[:train_num]
    test_data = data[train_num:]

    # shuffle
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)


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


def read_tsv_custom(path='/opt/ml/KorNLUDatasets/KorSTS/sts-dev.tsv'):
    column_names = {"genre":0, "filename":1, "year":2, "id":3, "score":4, "sentence1":5, "sentence2":6}
    lines = {v:[] for k,v in column_names.items()}
    check_idx = []
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            if idx != 0:
                line_val = line.rstrip().split('\t')
                if len(line_val) != 7:
                    check_idx.append(idx)
                else:
                    for idx, val in enumerate(line_val):
                        lines[idx].append(val)
    new_df = pd.DataFrame(lines)
    new_df = new_df.rename(columns={v:k for k,v in column_names.items()})
    new_df = new_df.rename(columns={'score':'labels'})

    print(f'passed {len(check_idx)} values')
    print(f'total : {len(new_df)}')
    return new_df

def preprocess_korSTS_data(data_path, train=True) -> DataFrameStr:
    # KorSTS train csv 데이터 경우 nan 처리 필요 -> 그냥 tsv로 읽기 통일
    # KorSTS tsv 파일은 split 오류 발생 -> custom 함수
    data = read_tsv_custom(data_path)

    # 중복 데이터 제거
    data.drop_duplicates(subset=['sent_a', 'sent_b'], inplace=True)
    # 데이터셋 갯수 확인
    if train:
        print('중복 제거 후 학습 데이터셋 : {}'.format(len(data)))
    else:
        print('중복 제거 후 테스트 데이터셋 : {}'.format(len(data)))

    # null 데이터 제거
    data.replace('', np.nan, inplace=True)
    data = data.dropna(how='any')

    if train:
        print('null 제거 후 학습 데이터셋 : {}'.format(len(data)))
    else:
        print('null 제거 후 테스트 데이터셋 : {}'.format(len(data)))

    return data

def preprocess_basic(path, train=True):
    data = pd.read_csv(path)
    # 중복 데이터 제거
    data.drop_duplicates(subset=['sent_a', 'sent_b'], inplace=True)
    # 데이터셋 갯수 확인
    if train:
        print('중복 제거 후 학습 데이터셋 : {}'.format(len(data)))
    else:
        print('중복 제거 후 테스트 데이터셋 : {}'.format(len(data)))

    # null 데이터 제거
    data.replace('', np.nan, inplace=True)
    data = data.dropna(how='any')

    if train:
        print('null 제거 후 학습 데이터셋 : {}'.format(len(data)))
    else:
        print('null 제거 후 테스트 데이터셋 : {}'.format(len(data)))

    return data


def get_label(df):
    try:
        return df['label'].values[0:]
    except:
        return df['labels'].values[0:]


def tokenizing_data(data, tokenizer,
                    data_type,
                    truncation=True,
                    max_length=64):
    if data_type == 'klueSTS':
        sent1_name = 'sentence1'
        sent2_name = 'sentence2'
    else:
        sent1_name = 'sent_a'
        sent2_name = 'sent_b'
    tokenized_sentences = tokenizer(
        list(data[sent1_name][0:]),
        list(data[sent2_name][0:]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=truncation,
        max_length=max_length
    )
    return tokenized_sentences