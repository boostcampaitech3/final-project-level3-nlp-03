

def compute_final_score(key_score:float , sim_score:float , alpha=0.5, mode='continuous',
                        sim_upper_threhold=0.8, sim_low_threshold=0.2):
    if sim_score <= 0:
        # cos similarity 경우 음의 값으로도 측정됨
        sim_score = 0
    assert 0<= key_score <=1 and 0<=sim_score<=1
    # 0<=key_score+ sim_score <=1 로 만들기 위해
    if mode == 'continuous':
        half_key = key_score
        half_sim = sim_score
        # 기본은 0.5 비율로 주는 것으로
        return round(half_key * (1-alpha) + half_sim*alpha,2)

    elif mode == 'step':
        half_key = key_score
        if sim_score >= sim_upper_threhold:
            # 어느정도 이상일 때는 만점 처리
            sim_score = 0.5

        elif sim_score < sim_low_threshold:
            # 완전 이상한 답일 때는 0점
            sim_score = 0

        else:
            sim_score = sim_score

    return half_key * (1-alpha) + sim_score*alpha

if __name__=='__main__':
    vals = compute_final_score(0.8,0.8, mode='continuous')
    print(vals)
    vals = compute_final_score(0.8, 0.8, mode='step')
    print(vals)
