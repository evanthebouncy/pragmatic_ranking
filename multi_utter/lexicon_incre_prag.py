from utils import *
from rect_incre_prag_1 import round_trip, get_global_rect_id, get_global_utt_id

import numpy as np
import matplotlib.pyplot as plt
import csv 

# make a valid lexicon
# take size of utterance and hypothesis
def make_lexicon(u_size, w_size, p=0.5):
    
    # check if lexicon is valid - non-empty
    def check_valid_non_empty(lexicon):
        # check if there is a 1 in each row
        for i in range(u_size):
            if np.sum(lexicon[i]) == 0:
                return False
        # check if there is a 1 in each column
        for i in range(w_size):
            if np.sum(lexicon[:, i]) == 0:
                return False
        return True
    
    # check if lexicon is valid - distinct rows and columns
    def check_valid_distinct(lexicon):
        # check if there are any duplicate rows
        if len(np.unique(lexicon, axis=0)) != u_size:
            return False
        # check if there are any duplicate columns
        if len(np.unique(lexicon.transpose(), axis=0)) != w_size:
            return False
        return True

    # make a random lexicon
    while True:
        # genearate a random lexicon with 0 1 entries where p(1) = p.
        lexicon = np.random.choice([0, 1], size=(u_size, w_size), p=[1-p, p])
        # lexicon = np.random.randint(2, size=(u_size, w_size))
        if check_valid_non_empty(lexicon) and check_valid_distinct(lexicon):
            return lexicon
        
if __name__ == '__main__':

    # make csv header row
    header_info = ['n_utter', 'lexicon_id', 'pair', 'success_rate']
    rest_rows = []
    # do it for rectangle like
    # U_size = 60
    # W_size = 200
    # prob_ = 0.5
    # utter_size_max = 6

    U_size = 30
    W_size = 100
    prob_ = 0.2
    utter_size_max = 3

    for lexicon_id in range(1,100):
        lexicon = make_lexicon(U_size, W_size, p=prob_)
        # invert the lexicon
        lexicon_inv = 1 - lexicon
        # concatenate the lexicon and its inverse, stack them vertically
        lexicon = np.vstack((lexicon, lexicon_inv))
        L0 = make_L0(lexicon)
        S0 = make_S0(lexicon)
        S1 = make_S1(lexicon)
        L1 = make_L1(lexicon)

        utt_ranks = get_global_utt_id(lexicon, n_iters = 100000)
        hyp_ranks = get_global_rect_id(lexicon, n_iters = 100000)
        Sr = make_ranking_S(lexicon, utt_ranks)
        Lr = make_ranking_L(lexicon, hyp_ranks)


        for n_utter in range(1, utter_size_max+1):
            print ("doing it for", n_utter)
            s0_l0 = round_trip(lexicon, S0, L0, n_utter)
            s0_l1 = round_trip(lexicon, S0, L1, n_utter)
            s1_l0 = round_trip(lexicon, S1, L0, n_utter)
            s1_l1 = round_trip(lexicon, S1, L1, n_utter)
            s1_lr = round_trip(lexicon, S1, Lr, n_utter)
            sr_l0 = round_trip(lexicon, Sr, L0, n_utter)
            sr_l1 = round_trip(lexicon, Sr, L1, n_utter)
            sr_lr = round_trip(lexicon, Sr, Lr, n_utter)
            print (f'n_utter {n_utter} s0_l0 {s0_l0} s0_l1 {s0_l1} s1_l0 {s1_l0} s1_l1 {s1_l1} s1_lr {s1_lr} sr_l0 {sr_l0} sr_l1 {sr_l1} sr_lr {sr_lr}')
            rest_rows.append([n_utter, lexicon_id, 's1_l1', s1_l1])
            rest_rows.append([n_utter, lexicon_id,'s1_lr', s1_lr])
            rest_rows.append([n_utter, lexicon_id,'s1_l0', s1_l0])
            rest_rows.append([n_utter, lexicon_id,'sr_l1', sr_l1])
            rest_rows.append([n_utter, lexicon_id,'sr_lr', sr_lr])
            rest_rows.append([n_utter, lexicon_id,'sr_l0', sr_l0])
            rest_rows.append([n_utter, lexicon_id,'s0_l0', s0_l0])
            rest_rows.append([n_utter, lexicon_id,'s0_l1', s0_l1])

    with open('lexicon_plot.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header_info)
        writer.writerows(rest_rows)