import pandas as pd
import re

def basic_checker(student_answers, key_words):
    val_dict = {
        'keyword': [],
        'start_idx': [],  # 같은 단어라도 여러번 나올 수 있음
        'end_idx': []
    }
    if pd.isnull(student_answers):
        return val_dict

    else:
        for key in key_words:
            results = re.finditer(key, student_answers)
            start_idx_list = []
            end_idx_list = []

            ### 단어가 여러번 나옴
            ## - start_idx, end_idx 는 나오는 순서에 따라 리스트
            ## - keyword는 unique한 값으로 리스트
            for found_key_idx, matched_key in enumerate(results):
                start_idx = matched_key.start()
                end_idx = matched_key.end()
                keys = student_answers[start_idx:end_idx]
                start_idx_list.append(start_idx)
                end_idx_list.append(end_idx)

                if found_key_idx == 0:
                    val_dict['keyword'].append(keys)
            val_dict['start_idx'].append(start_idx_list)
            val_dict['end_idx'].append(end_idx_list)

        return val_dict

def checker(student_answer, key_words):
    # 디테일한 사항은 구현 필요
    # 본 함수는 키워드가 주어졌을 때 정말 그 키워드 유무로만 비교하는 코드입니다.
    # key_words = ['대기', '공기']
    keyword_info_dict = basic_checker(student_answer, key_words)

    total_key_words = len(key_words)
    correct = len(keyword_info_dict['keyword']) # keyword가 여러번 나와도 한번만 카운팅
    matching_key_info= {
        'keyword': keyword_info_dict['keyword'],
        'start_idx' : keyword_info_dict['start_idx'],
        'end_idx' : keyword_info_dict['end_idx']
    }
    # 아웃풋 예시
    # 공기 1번, 따뜻이 2번 나온 경우
    # {'keword': ['공기', '따뜻'], 'start_idx': [[4], [0, 57]], 'end_idx': [[6], [2, 59]]}
    # 공기 1번, 따뜻 0번 나온 경우
    # {'keword': ['공기'], 'start_idx': [[4], []], 'end_idx': [[6], []]}
    # 아무것도 나오지 않은 경우
    # {'keword': [], 'start_idx': [], 'end_idx': []}
    ratio = 0
    if(total_key_words):
        ratio = round(correct/total_key_words, 2) 

    

    return ratio , matching_key_info