import torch
import json
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from keyword_checker import checker

from numpy import dot
from numpy.linalg import norm


def load_refine_json_data_multi(data):
    student_id = []
    answers = []
    for student_pair in data["answers"]:
        student_id.append(student_pair[0])
        answers.append(student_pair[1])

    # answers = pd.read_csv("./example.csv")["answer"].to_list()  # example
    # tmp. 실제 json 데이터 찾아서 읽어야 함
    # 제품의 가격이 낮아지고, 품질이 올라간다(좋아진다, 높아진다). 또 제품의 다양성이 증가하고, 소비자들은 더 좋은 혜택을 받을 수 있다.
    multi_input = [' 제품의 가격이 낮아지고, 품질이 올라간다. 또 제품의 다양성이 증가하고, 소비자들은 더 좋은 혜택을 받을 수 있다.',
                   '품질이 좋아지고 소비자들은 다양한 제품을 선택할 수 있다',
                   '소비자들은 다양한 혜택을 얻을 수 있고 더 싼 가격으로 제품을 구매할 수 있다.']

    gold_answer = multi_input
    # gold_answer = [data["gold_answer"]] * len(answers)
    return student_id, answers, gold_answer


def make_problem_df(problem, problem_idx, sim_score, student_id, answers):
    new_data = {}
    new_data["problem_idx"] = problem_idx
    new_data["question"] = problem["question"]
    new_data["gold_answer"] = problem["gold_answer"]
    new_data["keywords"] = problem["keywords"]

    result = []
    result_dict = {}
    result_len = len(student_id)
    for i in range(result_len):
        result_dict = {}
        check_score, match_info = checker(answers[i], new_data["keywords"])
        result_dict["student_id"] = student_id[i]
        result_dict["answer"] = answers[i]
        result_dict["sim_score"] = round(sim_score[i].astype(np.float64), 4)
        result_dict["keyword_score"] = check_score
        result_dict["total_score"] = round(sim_score[i] + check_score, 4)
        result_dict["match_info"] = match_info
        result.append(result_dict)

    new_data["result"] = result
    return new_data



def get_similarity(ans, right_ans, use="cosine"):
    # Cosine Similarity
    if use == "cosine":
        return dot(ans, right_ans) / (norm(ans) * norm(right_ans))

    # Euclidean
    if use == "euclidean":
        if norm(ans - right_ans) == norm(ans - right_ans):
            return norm(ans - right_ans)
        else:
            return -1

    # Pearson
    if use == "pearson":
        return dot((ans - np.mean(ans)), (right_ans - np.mean(right_ans))) / (
                    (norm(ans - np.mean(ans))) * (norm(right_ans - np.mean(right_ans))))


def sentences_sbert_predict_multi(emb_a, emb_b):
    total_results = []
    for student_emb in emb_b:
        results = []
        for gold_emb in emb_a:
            sim_score = get_similarity(gold_emb, student_emb, use="cosine")
            results.append(sim_score)
        total_results.append(round(sum(results)/len(results),2))
    return total_results

def inference_sbert_model_multi(data):
    LOAD_FROM = 'kimcando/ko-paraKQC-demo2'
    sbert_model = SentenceTransformer(LOAD_FROM)

    sbert_model.cuda()

    output_dict = {}

    subject = data["subject"]  # 과목
    output_dict["subject"] = data["subject"]
    new_problem = []
    # gold_answer = ['제과점끼리 경쟁 심화가 커질 수 있다.', '맛이 더 좋아질 수는 있따']
    # answers = ['제과점끼리 경쟁이 작아질 수 있다.', '더 좋은 맛을 누릴 수 있다'] # 경쟁이 커질 수 있다로 하면 낮게나옴
    for problem_idx, problem in enumerate(data["problem"]):
    # for i, problem in enumerate([1,3]):
        # for i, problem in range(data["problem"][0]):
        student_id, answers, gold_answer = load_refine_json_data_multi(problem)

        right_ans_emb = sbert_model.encode(gold_answer)
        stu_ans_emb = sbert_model.encode(answers)

        sim_score = sentences_sbert_predict_multi(right_ans_emb, stu_ans_emb)
        individual_df = make_problem_df(problem, problem_idx, sim_score, student_id, answers)
        new_problem.append(individual_df)

    output_dict["problem"] = new_problem
    output_json = json.dumps(output_dict)

    with open("./result_multi.json", "w") as f:  # result 눈으로 확인하는 용도
        json.dump(output_dict, f, ensure_ascii=False, indent=4)
    return output_json

def output_sbert():
    with open("./example.json", "r") as f:
        json_data = json.load(f)
    inference_sbert_model_multi(json_data)


if __name__=='__main__':
    # inference_sbert_model(None)
    output_sbert()

