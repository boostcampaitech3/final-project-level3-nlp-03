# preprocessing for SUBJ
import pandas as pd
from tqdm import tqdm
# 해당 과정은 crawling.py 이후의 preprocessing 과정입니다.
# 즉, 추출해야하는 명사, 형용사&동사의 단어를 네이버 사전으로부터 의미를 가져오고,
# [명사], [형용사] & [동사] 에 포함되는 단어들로만 추려서 저장된 dataframe을 사용합니다.

def filter_sub(df, noun_type):
    """
    df : crawling DataFrame
    noun_type: 명사의 어떤 noun type을  사용하는지 명시. 파일 저장 용도로 사용됨
    """

    v2_dict = {'v2': []}
    for desc in df['v1']:
        # not noun이면 처리 안함
        if desc == 'Not noun':
            v2_dict['v2'].append('Not noun')
            continue

        # 여러 개의 뜻을 가지는 경우 존재
        # 가장 우선순위 뜻만 사용할 것임으로 번호로 표기될 경우 "1." 로 1번 뜻만 가져옴
        if desc.startswith('1.'):
            out = desc.split('1.')[1].split('.')[0]
            # print('1 case: ', out)
        elif desc.startswith('['):
            out = desc.split(']')[1].split('.')[0]
            # print('bracket case: ', out)
        else:
            out = desc.lstrip()

        # 길이 길 경우 제거. 크롤링 할 때 길이 긴 설명은 페이지에 들어가야하기 때문
        if len(out) > 57:
            v2_dict['v2'].append('Longer')
        else:
            v2_dict['v2'].append(out)

    # 유의어 처리
    v3_dict = {'v3': []}
    for desc in v2_dict['v2']:
        # not noun이면 처리 안함
        if desc in ['Not noun', 'Longer']:
            v3_dict['v3'].append(desc)
            continue

        # 명사에 대해서는 유의어 처리를 일단 하지 않음 -> 이미 명사 너무 많음
        if '[유의어]' in desc:
            out = desc.split('[유의어]')[0]
        else:
            out = desc
        v3_dict['v3'].append(out)

    # 시작 단어에 특수문자 포함될 경우 제거
    v4_dict = {'v4': []}
    for desc in v3_dict['v3']:
        # not noun이면 처리 안함
        if desc in ['Not noun', 'Longer']:
            v4_dict['v4'].append(desc)
            continue
        if '<' in desc or '→' in desc:
            v4_dict['v4'].append('Start with special character')
        else:
            out = desc
            v4_dict['v4'].append(out)

    new_df = pd.concat([df, pd.DataFrame(v2_dict), pd.DataFrame(v3_dict), pd.DataFrame(v4_dict)], axis=1)
    new_df.to_csv(f'./preprocessed_{noun_type}.csv')
    print(f'{noun_type} done!')

def process_sim_words(word_lists):
    split_words = word_lists.split(',')
    new_words= []
    for word in split_words:
        if word[-1].isdigit():
            new_words.append(word[:-1])
        else:
            new_words.append(word)
    return new_words

def filter_verb(df, verb_type):
    """
    df : crawling DataFrame
    noun_type: 명사의 어떤 noun type을  사용하는지 명시. 파일 저장 용도로 사용됨
    """

    v2_dict = {'v2': []}
    for desc in df['v1']:
        # not noun이면 처리 안함
        if desc == 'Not verb':
            v2_dict['v2'].append(desc)
            continue

        # 1. 일 경우 1번 뜻만 가져온다
        if desc.startswith('1.'):
            out = desc.split('1.')[1].split('.')[0]
            print('1 case: ', out)
        elif desc.startswith('['):
            out = desc.split(']')[1].split('.')[0]
            print('bracket case: ', out)
        else:
            out = desc.lstrip()
        # 길이 길 경우 제거. 크롤링 할 때 길이 긴 설명은 페이지에 들어가야하기 때문
        if len(out) > 57:

            v2_dict['v2'].append('Longer')
        else:
            v2_dict['v2'].append(out)

    # 유의어 처리
    v3_dict = {'v3': [], '유의어':[]}
    for desc in v2_dict['v2']:
        # not noun이면 처리 안함
        if desc in ['Not verb', 'Longer']:
            v3_dict['v3'].append(desc)
            v3_dict['유의어'].append('NA')
            continue
        if '[유의어]' in desc:
            out = desc.split('[유의어]')[0]
            sim_words = desc.split('[유의어]')[1]
            # 동사의 경우 유의어 column을 따로 만듭니다.
            sim_words = process_sim_words(sim_words)
        else:
            out = desc
            sim_words = 'NA'
        v3_dict['v3'].append(out)
        v3_dict['유의어'].append(sim_words)

    # 시작 단어에 특수문자 포함될 경우 제거
    v4_dict = {'v4': []}
    for desc in v3_dict['v3']:
        # not noun이면 처리 안함
        if desc in ['Not verb', 'Longer']:
            v4_dict['v4'].append(desc)
            continue
        if "'" in desc or '→' in desc:
            v4_dict['v4'].append('Start with special character')
        else:
            out = desc
            v4_dict['v4'].append(out)

    new_df = pd.concat([df, pd.DataFrame(v2_dict), pd.DataFrame(v3_dict), pd.DataFrame(v4_dict)], axis=1)
    new_df.to_csv(f'./preprocessed_{verb_type}.csv')
    print(f'{verb_type} done!')

if __name__=='__main__':
    noun_type = ['NNP', 'NNG']
    for noun in noun_type:
        sub_df_path = f'/opt/ml/projects/notebooks/crawling_{noun}.csv'
        sub_df = pd.read_csv(sub_df_path)
        filter_sub(sub_df, noun)

    verb_type = ['VV', 'VA']
    for verb in verb_type:
        verb_df_path = f'/opt/ml/projects/notebooks/crawling_{verb}.csv'
        verb_df = pd.read_csv(verb_df_path)
        filter_verb(verb_df, verb)


