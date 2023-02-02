import random 
import numpy as np 

def make_L0(lexicon):
    def L0(utts):
        rows = [lexicon[u] for u in utts]
        raw_bits = np.prod(rows, axis=0)
        # normalize it
        return raw_bits / np.sum(raw_bits)
    return L0

def make_S0(lexicon):
    def S0(rect_id, n_utts):
        legal_utters = lexicon[:, rect_id]
        # get non-zero idexes to utts_col
        non_zero_ids = np.nonzero(legal_utters)[0]
        # sample n_utts from non_zero_ids
        replacement = False if n_utts < len(non_zero_ids) else True
        if replacement:
            return random.choices(non_zero_ids.tolist(), k=n_utts)
        return random.sample(non_zero_ids.tolist(), n_utts)
    return S0

def make_incre_S1(lexicon):
    # internal L0 for theory of mind
    L0 = make_L0(lexicon)
    def incre_S1(rect_id, prev_utts):
        utts_col = lexicon[:, rect_id]
        # get non-zero idexes to utts_col
        non_zero_ids = np.nonzero(utts_col)[0]
        all_utts_prob_unnorm = np.zeros(lexicon.shape[0])
        for u_id in non_zero_ids:
            extended_utts = prev_utts + [u_id]
            l0_response = L0(extended_utts)
            all_utts_prob_unnorm[u_id] = l0_response[rect_id]
        # normalize it
        all_utts_prob = all_utts_prob_unnorm / np.sum(all_utts_prob_unnorm)
        return all_utts_prob
    return incre_S1

def make_S1(lexicon, sample=False):
    incre_S1 = make_incre_S1(lexicon)
    def S1(rect_id, n_utts):
        utts_sofar = []
        for i in range(n_utts):
            utts_probs = incre_S1(rect_id, utts_sofar)
            if sample:
                u_id = np.random.choice(lexicon.shape[0], p=utts_probs)
            else:
                u_id = np.argmax(utts_probs)
            utts_sofar.append(u_id)
        return utts_sofar
    return S1

def make_S1_logprob(lexicon):
    incre_S1 = make_incre_S1(lexicon)
    def S1_logprob(rect_id, given_utts):
        logpr = 0
        utts_sofar = []
        for u_id in given_utts:
            utts_probs = incre_S1(rect_id, utts_sofar)
            logpr += np.log(utts_probs[u_id])
            utts_sofar.append(u_id)
        return logpr
    return S1_logprob

def make_L1(lexicon):
    S1_logprob = make_S1_logprob(lexicon)
    L0 = make_L0(lexicon)
    def L1(utts):
        l0_prob = L0(utts)
        l1_prob = np.zeros(lexicon.shape[1])
        for r_id in range(lexicon.shape[1]):
            if l0_prob[r_id] > 0:
                l1_prob[r_id] = np.exp(S1_logprob(r_id, utts))
        # normalize it
        return l1_prob / np.sum(l1_prob)
    return L1

def get_global_ord(items_to_rank, rank_list, n_iters_max = 1000):
    global_rank = items_to_rank
    swap_count = []
    swap_wait = 0
    for _ in range(1, n_iters_max):
        # if _ % 1000 == 0:
        #     # grab the last 1/4 of the swap_count
        #     if len(swap_count) > 0:
        #         last_fourth = swap_count[-int(len(swap_count)/4):]
        #         front, back = last_fourth[:int(len(last_fourth)/2)], last_fourth[int(len(last_fourth)/2):]
        #         # compare if their average is close
        #         if abs(np.mean(front) - np.mean(back)) / np.mean(front) < 0.1:
        #             print (front, back)
        #             print ('about converged after', _, 'iterations')
        #             break
        # pick a random row
        random_row = random.choice(rank_list)
        if len(random_row) < 2:
            continue
        # pick a pair
        random_pair = random.sample(list(random_row), 2)
        # get the index of the pair in ranked_row
        idx_in_ranked_row0, idx_in_ranked_row1 = [list(random_row).index(x) for x in random_pair]
        # get the index of the pair in ranked_all_utts
        idx_in_ranked_all_utts0, idx_in_ranked_all_utts1 = [global_rank.index(x) for x in random_pair]
        # check if the two pairs are the same order
        if idx_in_ranked_row0 < idx_in_ranked_row1 and idx_in_ranked_all_utts0 > idx_in_ranked_all_utts1:
            # swap them
            # print (f"{_} swapped")
            global_rank[idx_in_ranked_all_utts0], global_rank[idx_in_ranked_all_utts1] = \
                global_rank[idx_in_ranked_all_utts1], global_rank[idx_in_ranked_all_utts0]
            swap_count.append(swap_wait)
            swap_wait = 0
        else:
            swap_wait += 1
    return global_rank

def make_ranking_S(lexicon, utt_rank):
    def Sr(r_id, utters_number):
        legal_utts = lexicon[:, r_id]
        # get utters_number of utts in utt_rank that also matches the legal
        return [u for u in utt_rank if legal_utts[u] == 1][:utters_number]
    return Sr

def make_ranking_L(lexicon, hyp_rank):
    def Lr(utts):
        legal_hyps = np.prod(lexicon[utts], axis=0)
        # get the hyps in hyp_rank that also matches the legal
        return [h for h in hyp_rank if legal_hyps[h] == 1]
    return Lr