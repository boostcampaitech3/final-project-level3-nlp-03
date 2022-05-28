import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

#import ssl

NOT_ACCEPT = 0

def crawl_detail(href, category):
    detail = requests.get(href)
    detail_html = detail.content
    detail_soup = BeautifulSoup(detail_html, 'html.parser')
    detail_question = detail_soup.find('div', "c-heading__content")
    
    if detail_question is not None:
        detail_question = detail_question.text.strip()
    else:
        detail_question = "NaN"
    #content > div.question-content > div > div.c-heading._questionContentsArea.c-heading--default-old > div.c-heading__title > div > div

    detail_answer = detail_soup.find('div', "answer-content__list")
    
    accepted_idx = NOT_ACCEPT
    qa_list = [category, detail_question]
    for x in range(1, 4):
        answer = detail_answer.select_one(f'#answer_{x} > div.c-heading-answer__content > div.c-heading-answer__content-user')
        if answer is not None:
            qa_list.append(answer.text.strip())
        else:
            qa_list.append("NaN")
        
        accept = detail_answer.select_one(f'#answer_{x} > div.c-heading-answer > div.c-heading-answer__body > div.adopt-check-box')
        if accept is not None:
            accepted_idx = x

    qa_list.append(accepted_idx)
    return qa_list

'''
url -> 지식인 특정 카테고리에 Q&A를 긁는 기본 base url이다. 적어도 질문에 대한 답변이 하나는 존재한다
Q&A >교육,학문 > 중학교교육 > 사회  : 110302

'''
def crawl_kin(page_n: int, category_num: int):
    '''
    기본적으로 게시판에는 20개의 게시글이 있고 총 긁어올 정보는 20 x page_n만큼의 QA 정보입니다.
    category는 지식인의 어느 보드를 긁어올것인가에 대한 인자입니다.
    '''
    idx = 0
    web_df = pd.DataFrame(columns=('category', 'question', 'answer_1', 'answer_2', 'answer_3', 'accept'))
    detail_url = 'https://kin.naver.com/qna/detail.naver?'
    
    try:
        for page in tqdm(range(1, page_n+1)):
            url = f'https://kin.naver.com/qna/kinupList.naver?dirId={category_num}&page={page}'
            
            response = requests.get(url)
            if response.status_code == 200:
                html = response.text
                soup = BeautifulSoup(html, 'html.parser')
                ul = soup.find('tbody', id='au_board_list')
                tr_list = ul.findAll('tr')

                detail_list = []
                for id, tr_tag in enumerate(tr_list, start=1):
                    #한 게시판에 게시글이 20개씩있고 이것을 pagenation에 대해서 옮겨가야함
                    title_tag = tr_tag.select('td.title > a')[0]
                    category_tag = tr_tag.select('td.field > a')[0]
                    title = title_tag.text
                    detail_params = title_tag.get('href').split('?')[1]
                    category = category_tag.text
                    detail_href = detail_url + detail_params
                    detail_list.append((detail_href, category))

                for detail_href, category in detail_list:
                    web_df_column = crawl_detail(detail_href, category)
                    web_df.loc[idx] = web_df_column
                    idx += 1

            else:
                print(response.status_code)
                print(page)
                break
    except:
        pass


    web_df.to_csv(f"./{page_n * 20}_{category_num}.csv", index=False)


crawl_kin(100, 110304)