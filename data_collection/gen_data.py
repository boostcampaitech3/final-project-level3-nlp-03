## rule base generator
import random
import numpy as np

# 동작 파악을 위해 효율성보다 verbose하게 작성하였습니다.
class RuleBasedGenerator:
    TEMP1 = ['sub', '조사', 'verb']
    JOSA_LIST = ['은','는', '이', '가'] # 말이 안되더라도
    NEG_LIST = ['안', '아니', '못']
    def __init__(self, sub_df, verb_df, gen_pos_pair=2, gen_neg_pair=2, final_column_version='v4',antonym=True):
        self.sub_df = sub_df
        self.verb_df = verb_df
        self.final_column_version = final_column_version
        self.use_antonym = antonym
        self.gen_pos_pair = gen_pos_pair
        self.gen_neg_pair = gen_neg_pair

        self.sub_list = self.sub_df['word'].tolist()
        self.verb_list = self.verb_df['word'].tolist()
        self._set_random()

    def _set_random(self):
        random.seed(42)
        np.random.seed(42)

    def get_random_sub(self, sub_list=None):
        if sub_list is None:
            sub_list = self.sub_list

        while True:
            sub_idx = random.choice(range(len(sub_list)))
            if self.sub_df['v4'][sub_idx] != 'Not noun' and self.sub_df['v4'][sub_idx] != 'Longer' and \
                    self.sub_df['v4'][sub_idx] != 'Start with special character':
                break
            else:
                print('selecting sub again!')

        return sub_list[sub_idx], sub_idx

    def get_random_verb(self, verb_list=None):
        if verb_list is None:
            verb_list = self.verb_list

        while True:
            verb_idx = random.choice(range(len(verb_list)))
            if self.verb_df['v4'][verb_idx] != 'Not verb' and self.verb_df['v4'][verb_idx] != 'Longer' and \
                    self.verb_df['v4'][verb_idx] != 'Start with special character':
                break
            else:
                print('selecting verb again!')
        return verb_list[verb_idx], verb_idx

    def gen_pos_data(self, sub_w, sub_idx, verb_w, verb_idx):
        # '주어'의 단어를 뜻풀이로 변경
        total_list = []
        strs = ''
        for temp in self.TEMP1:
            if temp =='sub':
                strs += self.sub_df[self.final_column_version][sub_idx]

            elif temp =='조사':
                josa = random.choice(self.JOSA_LIST)
                strs += josa+' '

            elif temp=='verb':
                strs += verb_w
        total_list.append(strs)

        # '동사'의 단어를 뜻풀이로 변경
        strs = ''
        for temp in self.TEMP1:
            if temp == 'sub':
                strs += sub_w

            elif temp == '조사':
                josa = random.choice(self.JOSA_LIST)
                strs += josa + ' '

            elif temp == 'verb':
                strs += self.verb_df[self.final_column_version][verb_idx]
        total_list.append(strs)

        # TODO '동사의 유의어를 선택

        if not pd.isna(self.verb_df['유의어'][verb_idx]):
            strs = ''
            for temp in self.TEMP1:
                if temp == 'sub':
                    strs += sub_w

                elif temp == '조사':
                    josa = random.choice(self.JOSA_LIST)
                    strs += josa + ' '

                elif temp == 'verb':

                    strs += eval(self.verb_df['유의어'][verb_idx])[0] # 유의어 들 중 첫번째 단어가 그래도 제일 유사하니까 첫번째 유의어만 일단 선택
            total_list.append(strs)

        # 만들어진 Pair 중 설정한 값만큼만 반환
        return np.random.choice(total_list, self.gen_pos_pair,replace=False)

    def gen_neg_data(self, sub_w, sub_idx, verb_w, verb_idx):
        strs = ''
        total_list = []
        # 주어는 유지, 동사 앞에 부정부사 넣는 경우
        # 안, 아니 -, 못 - , -'다'제거하고 -기지 않기 때문이다./ -지 않기 때문이다.
        for temp in self.TEMP1:
            if temp == 'sub':
                strs += sub_w

            elif temp == '조사':
                josa = random.choice(self.JOSA_LIST)
                strs += josa + ' '

            elif temp == 'verb':
                neg = random.choice(self.NEG_LIST)
                strs += neg + ' ' + verb_w  # TODO :  -'다'제거하고 -기지 않기 때문이다./ -지 않기 때문이다.
        total_list.append(strs)

        # 주어를 뜻풀이로 바꾸고, 동사 앞에 부정부사 넣는 경우
        strs = ''
        for temp in self.TEMP1:
            if temp == 'sub':
                strs += self.sub_df[self.final_column_version][sub_idx]

            elif temp == '조사':
                josa = random.choice(self.JOSA_LIST)
                strs += josa + ' '

            elif temp == 'verb':
                neg = random.choice(self.NEG_LIST)
                strs += neg + ' ' + verb_w  # TODO :  -'다'제거하고 -기지 않기 때문이다./ -지 않기 때문이다.
        total_list.append(strs)

        # TODO 동사의 반의어 선택! 또는 다른 단어
        #antonym = False
        if self.use_antonym:
            if not pd.isna(self.verb_df['반의어'][verb_idx]):
                # breakpoint()
                strs = ''
                for temp in self.TEMP1:
                    if temp == 'sub':
                        strs += sub_w

                    elif temp == '조사':
                        josa = random.choice(self.JOSA_LIST)
                        strs += josa + ' '

                    elif temp == 'verb':
                        strs += self.verb_df['반의어'][verb_idx] # 유의어 들 중 첫번째 단어가 그래도 제일 유사하니까 첫번째 유의어만 일단 선택
                total_list.append(strs)

        # 만들어진 Pair 중 설정한 값만큼만 반환
        return np.random.choice(total_list, self.gen_neg_pair,replace=False)


    def make_pairs(self):
        # TODO : 단 한번도 뽑히지 않은 애들로만 뽑을 수 있게 하는 것

        sub_w, sub_idx = self.get_random_sub()
        verb_w, verb_idx = self.get_random_verb()
        org_sents = ''
        for temp in self.TEMP1:
            if temp == 'sub':
                org_sents += sub_w
            elif temp == 'verb':
                org_sents += verb_w
            elif temp=='조사':
                org_sents += '은' + ' ' # 기본형은 '은' 으로

        pos_pairs = self.gen_pos_data(sub_w, sub_idx, verb_w, verb_idx)
        neg_pairs = self.gen_neg_data(sub_w, sub_idx, verb_w, verb_idx)
        return {'sub_w':sub_w, 'verb_w':verb_w, 'org_sents':org_sents, 'pos_pairs':pos_pairs, 'neg_pairs':neg_pairs}

    def gen_data(self, num_itr=500, version_name='test'):
        org_data_dict = {'sub_w':[],
                           'verb_w':[],
                           'org_sents':[],
                           'pos_pairs':[],
                           'neg_pairs':[]}

        # 하지만 실제 데이터로 사용하려면..
        csv_data = {'sent_a':[],
                          'sent_b':[],
                          'labels':[],
                          'org_sents':[]} # org_sents는 식별 인자용
        for itr in range(num_itr):
            pair_out = self.make_pairs()
            org_sents = pair_out['org_sents']
            for k, v in org_data_dict.items():
                org_data_dict[k].append(pair_out[k])

            # 기본 조합  : org_sents + pos/ neg
            for pos_idx in range(len(pair_out['pos_pairs'])):
                csv_data['sent_a'].append(org_sents)
                csv_data['sent_b'].append(pair_out['pos_pairs'][pos_idx])
                csv_data['labels'].append(1)
                csv_data['org_sents'].append(org_sents)

            for neg_idx in range(len(pair_out['neg_pairs'])):
                csv_data['sent_a'].append(org_sents)
                csv_data['sent_b'].append(pair_out['neg_pairs'][neg_idx])
                csv_data['labels'].append(0)
                csv_data['org_sents'].append(org_sents)

            # 좀더 생각한 조합 : pos<->pos, neg<->neg, pos<->neg
            sent_a, sent_b = np.random.choice(pair_out['pos_pairs'], 2)
            csv_data['sent_a'].append(sent_a)
            csv_data['sent_b'].append(sent_b)
            csv_data['labels'].append(1)
            csv_data['org_sents'].append(org_sents)

            sent_a, sent_b = np.random.choice(pair_out['neg_pairs'], 2)
            csv_data['sent_a'].append(sent_a)
            csv_data['sent_b'].append(sent_b)
            csv_data['labels'].append(1)
            csv_data['org_sents'].append(org_sents)

            # 다른 pair에 대한 데이터가 더 만들어지게 될 것. -> 채점 입장에서 같은 것보다 얼마나 다르냐가 더 중요하니 그런 데이터를 더 모은다고 볼 수 있을까?
            for pos_idx in range(len(pair_out['pos_pairs'])):
                for neg_idx in range(len(pair_out['neg_pairs'])):
                    csv_data['sent_a'].append(pair_out['pos_pairs'][pos_idx])
                    csv_data['sent_b'].append(pair_out['neg_pairs'][neg_idx])
                    csv_data['labels'].append(0)
                    csv_data['org_sents'].append(org_sents)
        # breakpoint()
        df = pd.DataFrame(csv_data)
        df.to_csv(f'./gen_data_{version_name}_{len(csv_data)}.csv')


if __name__=='__main__':
    import pandas as pd
    # sub_1 = pd.read_csv('/opt/ml/projects/final-project-level3-nlp-03/data_collection/preprocessed_NNG.csv').drop(columns=['Unnamed: 0'])
    # sub_2 = pd.read_csv('/opt/ml/projects/final-project-level3-nlp-03/data_collection/preprocessed_NNP.csv').drop(columns=['Unnamed: 0'])
    #
    # verb_1 = pd.read_csv('/opt/ml/projects/final-project-level3-nlp-03/data_collection/preprocessed_VV.csv').drop(columns=['Unnamed: 0'])
    # verb_2 = pd.read_csv('/opt/ml/projects/final-project-level3-nlp-03/data_collection/preprocessed_VA.csv').drop(columns=['Unnamed: 0'])
    # sub_df = pd.concat([sub_1, sub_2],ignore_index=True)
    # verb_df = pd.concat([verb_1, verb_2],ignore_index=True)

  #  sub_1 = pd.read_csv('/opt/ml/projects/final-project-level3-nlp-03/data_collection/preprocessed_NNG.csv').drop(columns=['Unnamed: 0'])
    sub_2 = pd.read_csv('/opt/ml/projects/final-project-level3-nlp-03/data_collection/preprocessed_NNG_v5.csv').drop(columns=['Unnamed: 0'])

  #  verb_1 = pd.read_csv('/opt/ml/projects/final-project-level3-nlp-03/data_collection/preprocessed_VV.csv').drop(columns=['Unnamed: 0'])
    verb_2 = pd.read_csv('/opt/ml/projects/final-project-level3-nlp-03/data_collection/preprocessed_VA_v5.csv').drop(columns=['Unnamed: 0'])
  #  sub_df = pd.concat([sub_1, sub_2],ignore_index=True)
  #  verb_df = pd.concat([verb_1, verb_2],ignore_index=True)

    generator = RuleBasedGenerator(sub_2, verb_2,final_column_version='v5',antonym=True)
    generator.gen_data(num_itr=1800)
    print('Finished!')