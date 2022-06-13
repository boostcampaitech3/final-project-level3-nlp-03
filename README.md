# 안녕하세요, 곰파다입니다. 👋

> **곰파다** : 사물이나 일의 속내를 알려고 자세히 찾아보고 따지다.

> 최종 발표 [구글 슬라이드](https://docs.google.com/presentation/d/1bqOktYzIKhQZGDryHPhlE8sLUHnpPsBr/edit?usp=sharing&ouid=108719150340245142350&rtpof=true&sd=true) 및 [pdf](https://github.com/boostcampaitech3/final-project-level3-nlp-03/blob/main/NLP03_GOMPADA_Presentation.pdf) & [Youtube]()
> 
곰파다는 학생들의 서술형 답안을 자동으로 채점해 선생님들의 반복적 채점 작업을 효율적으로 줄여주는 프로젝트입니다. 교육기관에서 사용하는 AI 자동 채점 시스템이 기존의 단점(띄어쓰기, 유사어 등을 판별하지 못하는 것)으로 인해 실제로 사용하기 어렵다는 사실을 바탕으로, 단점들을 개선하고 더욱 정밀한 채점 보조 서비스를 만들고자 했습니다.


<img width="1000" alt="스크린샷 2022-06-08 오후 11 09 13" src="https://user-images.githubusercontent.com/50793789/172638023-73ea58e6-4a1f-448f-b9e5-9cf7f81cd9e8.png">

곰파다는 위와 같은 시스템 구조로 이루어져 있습니다. 사용자가 웹에서 문제, 모범답안, 키워드, 학생 답변 csv를 input으로 넣으면 곰파다가 데이터를 기반으로 학생의 답변을 자동으로 채점합니다. 해당 결과물은 사용자가 웹페이지에서 다운로드 받을 수 있는 형태인 CSV로 제공합니다. 




<center><img width="1028" alt="스크린샷 2022-06-08 오후 10 55 56" src="https://user-images.githubusercontent.com/50793789/172634752-4f18bd8c-d9ed-4e81-98bb-2a39d8a9f573.png"></center>
곰파다는 다섯 명의 협업으로 만들어진 프로젝트이며, 각자 맡았던 역할은 위와 같습니다.

---


## 기능 - 구현

### 모델

![image](https://user-images.githubusercontent.com/31491678/172570410-a4443871-6957-4999-912a-2b65014b49c1.png)
채점은 학생 답안과 모범답안과의 문장 유사도를 통한 점수와  문장 내의 키워드 점수를 통해 이루어집니다.

**1. 키워드 모델**

키워드 모델은 **Word2Vec을 사용**했으며 FastText를 기반으로 구현했습니다. 선생님이 작성한 키워드를 바탕으로 문장 내에서 **일정 수치 이상의 유사도 값을 가지는 단어들을 검출**하도록 구현했습니다. 그 결과 정확히 일치하는 단어 뿐만이 아니라 유사한 의미를 가진 단어들도 검출해 채점을 진행할 수 있었습니다.
정성적으로 판단한 결과 유사도 기준 수치(cosine similarity)를 0.35로 정했습니다.

![image](https://user-images.githubusercontent.com/31491678/172570441-3deac3e7-b850-4a3d-9238-b67250d004a1.png)

**2. 문장 유사도 모델**


문맥 유사도 채점 모델은 각 문장의 임베딩을 구하고 코사인 유사도 손실함수로 학습하는 **sentence BERT**를 이용했습니다. 일반적인 BERT모델로 문장쌍을 입력으로 넣고 회귀 혹은 이진분류로 학습하면 저희가 구축한 validation 데이터에서 좋지않은 성능이 나왔기 때문입니다. 더 좋은 문장 임베딩을 얻기 위해 사전 테스크와 다양한 데이터셋으로 실험해보았습니다.  그 결과 klue/bert-bas 사전학습 모델로 Natural langugage inference 테스크로 먼저 파인튜닝한 후,  문장 유사도 테스크에 파인튜닝한 모델을 선택하였습니다.

* klue/bert-base의 사전 학습 및 사용 데이터에 따른 성능 Ablations
<!-- <img width="934" alt="스크린샷 2022-06-08 오후 11 05 10" src="https://user-images.githubusercontent.com/50793789/172637099-9ba42056-59fa-4a78-8c2a-1645ea59377e.png">
 -->
 |     데이터셋    |               |                |                |                        |                              |     Threshold    |                |
|:---------------:|:-------------:|:--------------:|:--------------:|:----------------------:|:----------------------------:|:----------------:|:--------------:|
|      korNLI     |     korSTS    |     klueSTS    |     paraKQC    |     생성     데이터    |     korSTS  동사 일부 반의어 치환(500)    |        0.6       |       0.7      |
|                 |        ✓      |                |                |                        |                              |      0.60        |     0.59       |
|         ✓       |        ✓      |                |                |                        |                              |      0.67        |     0.65       |
|         ✓       |               |        ✓       |                |                        |                              |      0.63        |     0.58       |
|         ✓       |        ✓      |                |                |                        |                              |      0.53        |     0.61       |
|         ✓       |               |                |        ✓       |                        |                              |      0.64        |     0.59       |
|         ✓       |               |                |                |            ✓           |                              |      0.65        |     0.64       |
|         ✓       |        ✓      |                |                |                        |               ✓              |      0.52        |     0.52       |
<!-- <img width="960" height="300" alt="스크린샷 2022-06-08 오후 11 00 20" src="https://user-images.githubusercontent.com/50793789/172636095-61118d94-91e9-45e2-9c24-300f8f9521f0.png"> -->

* BERT(korSTS 회귀 학습)과 SBERT(klue/bert-base+NLI+STS) 의 문장 유사도 score 차이 예시 
<!-- <img width="1066" alt="스크린샷 2022-06-08 오후 11 02 12" src="https://user-images.githubusercontent.com/50793789/172636469-5835d781-6490-48f1-a658-473b2438bc13.png"> -->

|     모범답안    |     경쟁사가 있을 경우 서로의 상품이 잘 팔리게 하기 위해 가격도 낮추고 상품의 품질도 좋아진다. 또 상품의 다양성을 늘리는데도 도움이 된다.                    |     BERT    |     SBERT    |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|--------------|
|     답변 1            |     제품의 가격이 낮아지고,   품질이   좋아진다.   또   제품의 다양성이 증가하고, 소비자들은 더 좋은 혜택을 받을 수 있다.                                      |     0.56    |     0.81     |
|     답변 2            |     서로의 이권을 더 얻기 위해 품질이나 가격경쟁력 따위를 높이기 위해   노력하여 소비자는 더 질 좋으면서도 값싼 상품을 얻을 수 있을 것이다.                    |     0.52    |     0.75     |

| 모범      답안 | 고도가 높아지면 공기의 압력이 낮아지는데,   온도가   일정할 때 압력이 낮아지면 기체의 부피는 증가하므로 과자 봉지 내부 기체의 부피가 증가하기 때문이다. | BERT | SBERT |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|------|-------|
| 답변 1         | 기압이 낮아져서 공기의 압력 때문에 빵빵해지게 될 것일 것 같다.                                                                                          | 0.57 | 0.67  |
| 답변 2         | 평지에 비해 높은 산은 압력이 낮고 온도는 감소한다. PV=nRT에 따라 T는   감소하나, P의 감소 영향이 더 커서 부피는 증가하게 된다.                          | 0.51 | 0.66  |

* 짧은 문장에 SBERT 스코어 예시
    * 짧은 문장에 대한 반의어, 부정표현, 어순 변화는 파악하나 중요한 특정 단어가 등장하지 않을 경우 어려워하는 경향성 존재 
    
| 모범      답안 | 고도가 높아지면 공기의 압력이 낮아지는데,   온도가   일정할 때 압력이 낮아지면 기체의 부피는 증가하므로 과자 봉지 내부 기체의 부피가 증가하기 때문이다. | SBERT |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| 답변 1         | 지구는 주변을 돌지 않는다.               | 0.58 | 
| 답변 2         | 주변은 지구를 돈다.                    | 0.97 | 
| 답변 3         | 지구는 멈춰있다.                    | 0.54 | 
| 답변 4         | 멈춰있지 않고 중심을 기준으로 계속 움직인다                    | 0.47 | 
| 답변 5         | 지구는 멈춰있지 않고 중심을 기준으로 계속 움직인다                    | 0.76 | 

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

- 문장 유사도 채점 모델 Train 데이터(오픈 데이터)
  | |[KorSTS](https://github.com/kakaobrain/KorNLUDatasets)|[paraKQC](https://github.com/warnikchow/paraKQC)|[Kor-sentence](https://github.com/yoongi0428/Kor-Sentence-Similarity)|[KLUE STS](https://klue-benchmark.com/tasks/67/data/description)|
  |--|------|------|------|------|
  |라벨|0~5값|0 또는 1|0 또는 1|0 또는 1 / 0~5값|0 또는 1|
  |특징|짧은 문장.외국 STS-B 번역. 뉴스, 표현 설명 내용|짧은 문장. 질문중심|짧은 문장. 지식인 질문 포함. 인터넷 용어 다수| 짧은 문장. Airbnb, Policy, paraKQC 포함|
  |데이터 개수| 5,749 | 15,170 | 61,220 | 11,668 | 14,390|

- 문장 유사도 채점 모델 Train 데이터(제작 데이터)
    - 짧은 문장. 유의어, 반의어 고려해 데이터 제작
    - 라벨 : 0 또는 1
    - 데이터 개수 : 14,390
    
- 키워드 채점 & 문장 유사도 채점 모델 Validation & Test 데이터 : [교육부와 한국과학창의재단이 지원한 서술형 평가 지원프로그램 개발 사업 데이터(연구책임: 하민수 교수)](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002502290)
    - 초, 중, 고등학교 교육과정에 해당되는 사회, 과학 서술형 문제 64문항
    - [raw data](https://github.com/boostcampaitech3/final-project-level3-nlp-03/tree/main/data_validation/wai_raw_data) : 문제, 학생 답안, 문제에 대한 키워드 및 답안별 키워드 포함여부로 구성되어 있습니다. 
    - [validation data](https://github.com/boostcampaitech3/final-project-level3-nlp-03/blob/main/data_validation/validation_data_construction/validation.csv) : 실제로 모델 평가에 사용되었던 validation dataset입니다. 모범 답안 제작 및 파일럿 태깅을 통해 데이터를 직접 구축했습니다. 
    - [paired data](https://github.com/boostcampaitech3/final-project-level3-nlp-03/tree/main/data_validation/validation_data_construction/cosine_similarity) : raw data에 대해 모범 답안을 문제 별로 직접 제작하고, (모범 답안, 학생 답안) pair를 만들어 두 문장 임베딩 간의 cosine similarity를 구해서 pair 데이터셋을 제작했습니다. 
    


## 사용 매뉴얼
- 다음 매뉴얼을 참고해주세요
    **Gompada [사용 매뉴얼](https://www.notion.so/6e830bd4d4b1490692312a57b18942a5)  << Click**
    

### 협업 Tools

- 노션
    - 회의 기록을 포함해 서비스 구상도, 개발 과정 등 각종 공유 사안들을 체계적으로 정리했습니다.
    - 아래 링크를 통해 노션을 직접 확인하실 수 있습니다.
    - [notion](https://www.notion.so/bb2336eeb90040058b183835c34f4006)
    
      <img width="494" alt="스크린샷 2022-06-08 오후 11 12 15" src="https://user-images.githubusercontent.com/50793789/172638642-0c96621f-abf9-4d2e-b788-c9947ccfd54e.png">

- GitHub
![image](https://user-images.githubusercontent.com/31491678/172570704-0ddb159d-d99c-41f1-b169-7ef8a821b6df.png)
- 개발 인원 별로 Branch를 나누어 Pull Reqeust 기반 협업을 진행했습니다.
- Commit과 관련된 내용을 작성해 공유하고자 했습니다.
![image](https://user-images.githubusercontent.com/31491678/172570738-cb17f5a5-2029-4196-b8da-e255b1a30d28.png)
