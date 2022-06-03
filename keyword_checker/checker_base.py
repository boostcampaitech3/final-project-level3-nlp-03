# lemmatization 적용하지 않았습니다!
# from konlpy.tag import Mecab
# from hanspell import spell_checker
# from pykospacing import Spacing
import re
import pandas as pd

def lemmatization(text):
    mecab = Mecab()
    # hanspell_result = spell_checker.check(text)
    # text = hanspell_result.checked
    return mecab.nouns(text)

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



if __name__=='__main__':
    ### 밑을 내리면 함수화하기 전의 테스트코드도 존재합니다.
    # 아래는 함수화하기 전의 테스트코드입니다.
    answer_csv = pd.read_csv('/opt/ml/projects/notebooks/checker_test_data.csv').drop(columns='Unnamed: 0')
    # 답이 없는 경우도 있지만, null처리가 되면 그 학생의 정보가 사라지기 때문에 유지는 해둬야함
    answer_dict = {answer_csv.student_id[idx]: None for idx in range(len(answer_csv))}

    key_words = ['공기', '따뜻']
    # null있어서 iterrows로 하면 곤란하다
    for idx in range(len(answer_csv)):
        student_id = answer_csv['student_id'].iloc[idx]
        val_dict = {
            'keyword': [],
            'start_idx': [],  # 같은 단어라도 여러번 나올 수 있음
            'end_idx': []
        }
        sent = answer_csv['answers'].iloc[idx]
        val_dict = basic_checker(sent, key_words)
        answer_dict[student_id] = val_dict
    breakpoint()
    print('')

    #### 아래는 함수화하기 전의 테스트코드입니다.
    """
    answer_csv = pd.read_csv('/opt/ml/projects/notebooks/checker_test_data.csv').drop(columns='Unnamed: 0')
    # 답이 없는 경우도 있지만, null처리가 되면 그 학생의 정보가 사라지기 때문에 유지는 해둬야함
    answer_dict = {answer_csv.student_id[idx]: None for idx in range(len(answer_csv))}

    key_words = ['공기', '따뜻']
    # null있어서 iterrows로 하면 곤란하다
    for idx in range(len(answer_csv)):
        student_id = answer_csv['student_id'].iloc[idx]
        val_dict = {
            'keyword': [],
            'start_idx': [],  # 같은 단어라도 여러번 나올 수 있음
            'end_idx': []
        }
        if pd.isnull(answer_csv['answers'].iloc[idx]):
            answer_dict[student_id] = val_dict
            continue
        student_answers = answer_csv['answers'].iloc[idx]

        for key in key_words:
            results = re.finditer(key, student_answers)
            start_idx_list = []
            end_idx_list = []
            ### 단어가 여러번 나옴
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

        answer_dict[student_id] = val_dict
    """


