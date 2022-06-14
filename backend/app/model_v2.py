import torch
import json
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

from numpy import dot
from numpy.linalg import norm
from keyword_similarity import make_keyword_list


def load_refine_json_data(data):
    student_id = []
    answers = []
    for student_pair in data["answers"]:
        student_id.append(student_pair[0])
        answers.append(student_pair[1])

    gold_answer = [data["gold_answer"]] * len(answers)
    return student_id, answers, gold_answer
#여기도 부를것
def sync_match_info(answer, keywords, keyword_num, whole_keyword_list, student_num):
    start_idx_list = []
    end_idx_list = []
    word_list = []
    match_info = {}
    match_info["keyword"] = []
    match_info["similarity_keyword"] = []
    match_info["start_idx"] = []
    empty_count = 0


    #keyword의 길이
    len_keywords = len(keywords)

    for i in range(len_keywords):
        start_idx = whole_keyword_list[f"keyword_{i}"][f"student_{student_num}"][0][
            "start_idx"
        ]
        end_idx = whole_keyword_list[f"keyword_{i}"][f"student_{student_num}"][1][
            "end_idx"
        ]
        words = whole_keyword_list[f"keyword_{i}"][f"student_{student_num}"][2]["word"]
        start_idx_list.append(start_idx)
        end_idx_list.append(end_idx)
        word_list.append(words)
        if not start_idx:
            empty_count += 1
        else:
            match_info["keyword"].append(keywords[i])
    
    for i in range(len(match_info["keyword"])):
        if not match_info["keyword"][i] in answer:
            match_info["similarity_keyword"].append(match_info["keyword"][i])
    
    match_info["keyword"] = [word for word in match_info["keyword"] if word not in match_info["similarity_keyword"]]

    match_info["start_idx"] = start_idx_list
    match_info["end_idx"] = end_idx_list
    match_info["word"] = word_list

    keyword_score = (len_keywords - empty_count) / len_keywords

    return keyword_score, match_info


#여기 분를것
def make_problem_df(problem, problem_idx, sim_score, student_id, answers):
    new_data = {}
    new_data["problem_idx"] = problem_idx
    new_data["question"] = problem["question"]
    new_data["gold_answer"] = problem["gold_answer"]
    new_data["keywords"] = problem["keywords"]

    keyword_num, whole_keyword_list = make_keyword_list(new_data["keywords"], answers)

    result = []
    result_dict = {}
    result_len = len(student_id)
    for i in range(result_len):
        result_dict = {}
        keyword_score, match_info = sync_match_info(answers[i], new_data["keywords"], keyword_num, whole_keyword_list, i)

        result_dict["student_id"] = student_id[i]
        result_dict["answer"] = answers[i]
        result_dict["sim_score"] = round(sim_score[i].astype(np.float64), 4)
        result_dict["keyword_score"] = keyword_score
        result_dict["total_score"] = round(sim_score[i] + keyword_score, 4)
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


def sentences_sbert_predict(emb_a, emb_b):
    results = []
    for idx, (a, b) in enumerate(zip(emb_a, emb_b)):
        sim_score = get_similarity(a, b, use="cosine")
        results.append(round(sim_score,2))
    return results

def inference_sbert_model(data):
    LOAD_FROM = 'kimcando/ko-paraKQC-demo2'
    sbert_model = SentenceTransformer(LOAD_FROM)

    sbert_model.cuda()
    output_dict = {}

    subject = data["subject"]  # 과목
    output_dict["subject"] = data["subject"]
    new_problem = []

    for problem_idx, problem in enumerate(data["problem"]):
        student_id, answers, gold_answer = load_refine_json_data(problem)

        right_ans_emb = sbert_model.encode(gold_answer)
        stu_ans_emb = sbert_model.encode(answers)
        sim_score = sentences_sbert_predict(right_ans_emb, stu_ans_emb)
        individual_df = make_problem_df(problem, problem_idx, sim_score, student_id, answers)
        new_problem.append(individual_df)

    output_dict["problem"] = new_problem
    output_json = json.dumps(output_dict)
    with open("./result_v2.json", "w") as f:  # result 눈으로 확인하는 용도
        json.dump(output_dict, f, ensure_ascii=False, indent=4)
    return output_json

def output_sbert():
    with open("./example.json", "r") as f:
        json_data = json.load(f)
    inference_sbert_model(json_data)


if __name__=='__main__':
    output_sbert()

