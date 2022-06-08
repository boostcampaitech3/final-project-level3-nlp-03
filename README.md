# 안녕하세요, 곰파다입니다. 👋

![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled.png)

> **곰파다** : 사물이나 일의 속내를 알려고 자세히 찾아보고 따지다.


곰파다는 학생들의 서술형 답안을 자동으로 채점해 선생님들의 반복적 채점 작업을 효율적으로 줄여주는 프로젝트입니다. 교육기관에서 사용하는 AI 자동 채점 시스템이 기존의 단점(띄어쓰기, 유사어 등을 판별하지 못하는 것)으로 인해 실제로 사용하기 어렵다는 사실을 바탕으로, 단점들을 개선하고 더욱 정밀한 채점 보조 서비스를 만들고자 했습니다.

![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%201.png)

곰파다는 위와 같은 시스템 구조로 이루어져 있습니다. 사용자가 웹에서 문제, 모범답안, 키워드, 학생 답변 csv를 input으로 넣으면 곰파다가 데이터를 기반으로 학생의 답변을 자동으로 채점합니다. 해당 결과물은 사용자가 웹페이지에서 다운로드 받을 수 있는 형태인 CSV로 제공합니다. 

![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%202.png)

곰파다는 다섯 명의 협업으로 만들어진 프로젝트이며, 각자 맡았던 역할은 위와 같습니다.

---


## 기능 - 구현

### 모델

![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%203.png)

채점은 학생 답안과 모범답안과의 문장 유사도를 통한 점수와  문장 내의 키워드 점수를 통해 이루어집니다.

1. 키워드 모델

키워드 모델은 **Word2Vec을 사용**했으며 FastText를 기반으로 구현했습니다. 선생님이 작성한 키워드를 바탕으로 문장 내에서 **일정 수치 이상의 유사도 값을 가지는 단어들을 검출**하도록 구현했습니다.

그 결과 정확히 일치하는 단어 뿐만이 아니라 유사한 의미를 가진 단어들도 검출해 채점을 진행할 수 있었습니다.

정성적으로 판단한 결과 유사도 기준 수치(cosine similarity)를 0.35로 정했습니다.

![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%204.png)

1. 문장 유사도 모델

문맥 유사도 채점 모델은 각 문장의 임베딩을 구하고 코사인 유사도 손실함수로 학습하는 **sentence BERT**를 이용했습니다. 일반적인 BERT모델로 문장쌍을 입력으로 넣고 회귀 혹은 이진분류로 학습하면 저희가 구축한 validation 데이터에서 좋지않은 성능이 나왔기 때문입니다. 더 좋은 문장 임베딩을 얻기 위해 사전 테스크와 다양한 데이터셋으로 실험해보았습니다.  그 결과 klue/bert-bas 사전학습 모델로 Natural langugage inference 테스크로 먼저 파인튜닝한 후,  문장 유사도 테스크에 파인튜닝한 모델을 선택하였습니다.

![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%205.png)

![마크다운으로 SBERT 실험표 추가 할듯](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%206.png)

마크다운으로 SBERT 실험표 추가 할듯

![마크다운으로 샘플 예시 추가할듯](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%207.png)

마크다운으로 샘플 예시 추가할듯

## 모델 환경 설정

1. pip install -r requirements.txt
2. sh 커맨드로 requirements.sh 설치

## Frontend

- 다음 매뉴얼을 참고해주세요
    - [Frontend](https://github.com/boostcampaitech3/final-project-level3-nlp-03/tree/main/frontend) / [README.md](https://github.com/boostcampaitech3/final-project-level3-nlp-03/blob/main/frontend/README.md)

## Backend

- 다음 매뉴얼을 참고해주세요
    - [Backend](https://github.com/boostcampaitech3/final-project-level3-nlp-03/tree/main/backend) / [README.md](https://github.com/boostcampaitech3/final-project-level3-nlp-03/blob/main/backend/README.md)

---

## 데이터

- 문장 유사도 Train 데이터
    
    ![스크린샷 2022-06-08 오전 9.36.24.png](ReadME%206bbc2437b908418bb5404eceba7358ee/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-06-08_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_9.36.24.png)
    
    [1] [https://github.com/kakaobrain/KorNLUDatasets](https://github.com/kakaobrain/KorNLUDatasets)
    
    [2] [https://github.com/warnikchow/paraKQC](https://github.com/warnikchow/paraKQC)
    
    [3] [https://github.com/yoongi0428/Kor-Sentence-Similarity](https://github.com/yoongi0428/Kor-Sentence-Similarity)
    
    [4] [https://klue-benchmark.com/tasks/67/data/description](https://klue-benchmark.com/tasks/67/data/description)
    
- 문장 유사도  Validation & Test 데이터
    - 교육부와 한국과학창의재단이 지원한 서술형 평가 지원프로그램 개발 사업 데이터(연구책임: 하민수 교수)

[데이터 리드미](https://www.notion.so/f98c957929bb4d1ea84c3909c1cd93d7)

## 사용 매뉴얼

- 다음 매뉴얼을 참고해주세요
    
    **Gompada [사용 매뉴얼](https://www.notion.so/6e830bd4d4b1490692312a57b18942a5)  << Click**
    

## Appendix

### 타임라인

타임라인은 아래 캘린더를 참고해주시면 됩니다.

- 개발 기간 동안 주중 아침 10시 그리고 오후 4시에 회의를 진행했으며 진행 사안들을 공유했습니다.

![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%208.png)

### 협업 Tools

- 노션
    - 회의 기록을 포함해 서비스 구상도, 개발 과정 등 각종 공유 사안들을 체계적으로 정리했습니다.
    - 아래 링크를 통해 노션을 직접 확인하실 수 있습니다.
    - [notion](https://www.notion.so/bb2336eeb90040058b183835c34f4006)

![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%209.png)

- GitHub
    
    ![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%2010.png)
    
- 개발 인원 별로 Branch를 나누어 Pull Reqeust 기반 협업을 진행했습니다.
- Commit과 관련된 내용을 작성해 공유하고자 했습니다.

![Untitled](ReadME%206bbc2437b908418bb5404eceba7358ee/Untitled%2011.png)
