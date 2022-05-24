import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_sub(word):
    url = f'https://dict.naver.com/search.dict?dicQuery={word}&query={word}&target=dic&ie=utf8&query_utf=&isOnlyViewEE='
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    try:
        sent = soup.select('p')[4].text
    except:
        return None
    replace = replace_all(sent, replace_list)
    final = replace.split('[명사]')
    # final = replace.split('[명사]')[1].split('1.')[1].split('.')[0]
    return final

def get_verb(word, mode=None):
    url = f'https://dict.naver.com/search.dict?dicQuery={word}&query={word}&target=dic&ie=utf8&query_utf=&isOnlyViewEE='
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    try:
        sent = soup.select('p')[4].text
    except:
        return None
    replace = replace_all(sent, replace_list)
    final = replace.split('[형용사]')
    # 형용사가 존재하면 길이가 2 
    if not len(final) ==2:
        print(final)
        final = final[0].split('[동사]')
    # final = replace.split('[명사]')[1].split('1.')[1].split('.')[0]
    return final

def replace_all(text, remove_list):
    for i, j in remove_list:
        text = text.replace(i, j)
    return text
replace_list = [('\t',''), ('\r',''),('\t',''), ('\n','')]

# noun first
base_path = './WAI-para_'
path_lists = ['NNP']
import os, sys
for path in path_lists:
    name = base_path+path + '.csv'
    nouns = pd.read_csv(name).drop(columns=['Unnamed: 0'])

    from tqdm import tqdm
    import time
    data_dict = {'v1':[]}
    for w_idx in tqdm(range(len(nouns[path]))):
        # print(nouns.NNG[w_idx])
        out = get_sub(nouns[path][w_idx])
        if out is None:
            out = ['NA']
            print(w_idx, nouns[path][w_idx], out[0])
        try:
            data_dict['v1'].append(out[1])
        except:
            data_dict['v1'].append('Not noun')
    df = pd.DataFrame(data_dict)
    df.to_csv(f'./crawling_{path}')

path_lists = ['VA', 'VV']
for path in path_lists:
    name = base_path+path + '.csv'
    
    verb = pd.read_csv(name).drop(columns=['Unnamed: 0'])
    data_dict = {'v1':[]}
    for w_idx in tqdm(range(len(verb[path]))):
        # print(nouns.NNG[w_idx])
        out = get_verb(verb[path][w_idx]) #한번에..
        if out is None:
            out = ['NA']
            print(w_idx, verb[path][w_idx], out[1])
        try:
            data_dict['v1'].append(out[1])
        except:
            print(verb[path][w_idx], out)
            data_dict['v1'].append('Not verb')
    df = pd.DataFrame(data_dict)
    name = path.split('/')[1]
    df.to_csv(f'./crawling_{path}')

    


