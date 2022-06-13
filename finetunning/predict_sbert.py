
import pandas as pd


# https://wikidocs.net/154530
import numpy as np
import pandas as pd
import urllib.request
from sentence_transformers import SentenceTransformer
# 'sentence-transformers/xlm-r-large-en-ko-nli-ststb'
# '/opt/ml/projects/final-project-level3-nlp-03/finetunning/output/paraKQC'
# sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens
# /opt/ml/projects/final-project-level3-nlp-03/finetunning/output/kor-sentence
# sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens
# '/opt/ml/projects/final-project-level3-nlp-03/finetunning/output/para+MRC-bin-ko'
sbert_kor = SentenceTransformer('/opt/ml/projects/final-project-level3-nlp-03/finetunning/output/klue-bert-nli_sts_cosine')

from numpy import dot
from numpy.linalg import norm


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


# %%

# pairs = []
# for i, emb in enumerate(embeddings_kor):
#     for j in range(i + 1, len(embeddings_kor)):
#         pairs.append((sentences[i], sentences[j], emb, embeddings_kor[j]))
# # pairs[0]
#
# for (sent1, sent2, ans, r_ans) in pairs:
#     print(sent1, " // ", sent2, get_similarity(ans, r_ans))

# %%

data_path = '/opt/ml/projects/tunning_data/demo_eval_v2.csv'
df_valid_v1 = pd.read_csv(data_path)
df_valid_v1

prediction = []
for data in df_valid_v1.iloc:
    right_ans = data['sent_a']
    student_ans = data['sent_b']
    label = data['labels']
    # print(right_ans, '//', student_ans, '//', label)

    right_ans_emb = sbert_kor.encode([right_ans])
    stu_ans_emb = sbert_kor.encode([student_ans])

    sim = get_similarity(right_ans_emb[0], stu_ans_emb[0], use="cosine")
    prediction.append(sim)

df_valid_v1['predict(cosine_sim)'] = prediction
# df_valid_v1


# %%

def threshhold_num(n=0.6):
    def threshold(pred):
        if pred > n:
            return 1
        elif pred <= n:
            return 0

    return threshold


df_valid_v1['>0.6'] = df_valid_v1['predict(cosine_sim)'].apply(threshhold_num(0.6))
df_valid_v1['>0.7'] = df_valid_v1['predict(cosine_sim)'].apply(threshhold_num(0.7))
df_valid_v1['>0.8'] = df_valid_v1['predict(cosine_sim)'].apply(threshhold_num(0.8))
df_valid_v1['>0.9'] = df_valid_v1['predict(cosine_sim)'].apply(threshhold_num(0.9))
# df_valid_v1.head(50)

# %%

df_valid_v1.to_csv('./inference_results/sbert/kosbert-klue-bert-base-cosine-demo_eval_v2.csv')

# %%
def get_acc(thresh):
    cnt = 0
    for idx in range(len(df_valid_v1)):
        if df_valid_v1['labels'][idx] == df_valid_v1[f'>{thresh}'][idx]:
            cnt += 1
    print(cnt / len(df_valid_v1))

get_acc(0.6)
get_acc(0.7)
get_acc(0.8)
get_acc(0.9)

