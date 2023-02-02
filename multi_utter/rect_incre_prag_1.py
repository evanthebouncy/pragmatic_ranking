# make the incremental version of pragmatics
import random 
from utils import get_global_ord, make_L0, make_S0, make_L1,\
make_incre_S1, make_S1, make_S1_logprob, make_ranking_S, make_ranking_L
import numpy as np 

def get_S1_data(lexicon):
    incre_S1 = make_incre_S1(lexicon)
    L0 = make_L0(lexicon)
    data = []
    all_rect_idx = list(range(lexicon.shape[1]))
    for r_id in all_rect_idx:
        last_valid_num_rect_size = lexicon.shape[1]
        utts_sofar_idx = []
        for n_utts in range(1, 100):
            nxt_utt_probs = incre_S1(r_id, utts_sofar_idx)
            # argmax the next utt
            nxt_utt = np.argmax(nxt_utt_probs)
            utts_sofar_idx.append(nxt_utt)
            # stops if we do not reduce the hypothesis space further
            # i.e. there's nothing added by new utterances
            l0_probs = L0(utts_sofar_idx)
            # check how many rects are still valid from l0_probs
            num_rect_size = np.count_nonzero(l0_probs)
            if num_rect_size == last_valid_num_rect_size:
                break
            # update the last valid num_rect_size
            last_valid_num_rect_size = num_rect_size
            # rank of the nxt_utt_probs, sorted in descending order
            rank = np.argsort(nxt_utt_probs)[::-1].tolist()
            # remove entry from rank where entry is 0
            rank = [r for r in rank if nxt_utt_probs[r] > 0]
            data.append(rank) 
    return data

def get_S1_L1_data(lexicon):
    S1 = make_S1(lexicon)
    L1 = make_L1(lexicon)
    L0 = make_L0(lexicon)
    data = []
    all_rect_idx = list(range(lexicon.shape[1]))
    for r_id in all_rect_idx:
        last_valid_num_rect_size = lexicon.shape[1]
        for n_utts in range(1, 100):
            utts_sofar_idx = S1(r_id, n_utts)
            # stops if we do not reduce the hypothesis space further
            # i.e. there's nothing added by new utterances
            l0_probs = L0(utts_sofar_idx)
            # check how many rects are still valid from l0_probs
            num_rect_size = np.count_nonzero(l0_probs)
            if num_rect_size == last_valid_num_rect_size:
                break
            # update the last valid num_rect_size
            last_valid_num_rect_size = num_rect_size
            l1_probs = L1(utts_sofar_idx)
            # get the ranks
            rank = np.argsort(l1_probs)[::-1].tolist()
            # remove the entries from rank where prob is 0
            rank = [x for x in rank if l1_probs[x] > 0]
            data.append((utts_sofar_idx, rank))

    return data

def get_global_utt_id(lexicon, n_iters = 100000):
    rank_list = get_S1_data(lexicon)
    all_utter_idxs = list(range(lexicon.shape[0]))
    global_utt_ranked = get_global_ord(all_utter_idxs, rank_list, n_iters)
    return global_utt_ranked

def get_global_rect_id(lexicon, n_iters = 100000):
    data = get_S1_L1_data(lexicon)
    all_rect_idxs = list(range(lexicon.shape[1]))
    ranked_list = [x[1] for x in data]
    global_rect_ranked = get_global_ord(all_rect_idxs, ranked_list, n_iters)
    return global_rect_ranked

def visualize_rank_utt(ranked_utts):
    import matplotlib.pyplot as plt
    plt.xlim(0, MAX_LEN)
    plt.ylim(0, MAX_LEN)
    # inverse the y axis
    plt.gca().invert_yaxis()
    # show the x axis labels from 1 to 10
    plt.xticks(range(0, MAX_LEN))
    # show the y axis labels from 1 to 10
    plt.yticks(range(0, MAX_LEN))
    # make the plot a square
    plt.gca().set_aspect('equal', adjustable='box')
    for i, u in enumerate(ranked_utts):
        coord, bool = u
        color = 'g' if bool else 'r'
        shift = 0 if bool else 0.2
        # write 'i' on the coordinate
        plt.text(coord[0]+shift, coord[1]+shift, str(i), color=color)
    plt.savefig('tmp/rect0_ord.png')
    plt.close()

def round_trip(lexicon, speaker, listener, n_utts):
    succ = 0
    total = lexicon.shape[1]
    for hyp in range(lexicon.shape[1]):
        utts = speaker(hyp, n_utts)
        hyp_rec_pr_or_rank = listener(utts)
        hyp_rec = None # tbd
        # if this is a probability distribution, i.e. has floats
        if 'dtype' in dir(hyp_rec_pr_or_rank) and hyp_rec_pr_or_rank.dtype == np.float64:
            hyp_rec = np.argmax(hyp_rec_pr_or_rank)
        # if this is a ranking, i.e. has ints
        else:
            hyp_rec = hyp_rec_pr_or_rank[0]
        if hyp == hyp_rec:
            succ += 1
    return succ / total

if __name__ == '__main__':

    from rect import *
    import csv 

    R, U, lexicon = get_lexicon()
    # print (len(R), len(U), lexicon.shape)
    # # print the numbers of 1s in lexicon
    # print (np.count_nonzero(lexicon) / lexicon.size)
    # assert 0
    L0 = make_L0(lexicon)
    S0 = make_S0(lexicon)
    S1 = make_S1(lexicon)
    L1 = make_L1(lexicon)

    # quickly visualize S1
    if False:
        r_id = np.random.randint(0, len(R)-1)
        utts_sofar_idx = S1(r_id, 3)
        utts_sofar = [U[u_id] for u_id in utts_sofar_idx]
        R[r_id].draw('tmp/rect_S1_utts.png', utts_sofar)
        assert 0

    utt_ranks = get_global_utt_id(lexicon, n_iters = 100000)
    rect_ranks = get_global_rect_id(lexicon, n_iters = 100000)
    Sr = make_ranking_S(lexicon, utt_ranks)
    Lr = make_ranking_L(lexicon, rect_ranks)

    # make csv header row
    header_info = ['n_utter', 'pair', 'success_rate']
    rest_rows = []
    for n_utter in [1,2,3,4,5,6]:
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
        rest_rows.append([n_utter, 's1_l1', s1_l1])
        rest_rows.append([n_utter, 's1_lr', s1_lr])
        rest_rows.append([n_utter, 's1_l0', s1_l0])
        rest_rows.append([n_utter, 'sr_l1', sr_l1])
        rest_rows.append([n_utter, 'sr_lr', sr_lr])
        rest_rows.append([n_utter, 'sr_l0', sr_l0])
        rest_rows.append([n_utter, 's0_l0', s0_l0])
        rest_rows.append([n_utter, 's0_l1', s0_l1])
        
    with open('rect_plot.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header_info)
        writer.writerows(rest_rows)
    
    # incre_S1 = make_incre_S1(lexicon)

    # r_id = np.random.randint(0, len(R)-1)
    # utters_number = 3

    # utts_probs = incre_S1(r_id, [])
    # # get the top 10 utts
    # top_utts_idx = np.argsort(utts_probs)[::-1][:utters_number]
    # top_utts = [U[u_id] for u_id in top_utts_idx]
    # R[r_id].draw('tmp/rect_ord_utts.png', top_utts)

    # global_ranked_utts = get_global_utt_id(lexicon, n_iters = 1000)
    # print (global_ranked_utts)

    # for xx in global_ranked_utts:
    #     print (U[xx])
    
    # visualize_rank_utt([U[u] for u in global_ranked_utts])

    # S1 = make_S1(lexicon, sample=False)
    # utts_sofar_idx = S1(r_id, utters_number)
    # utts_sofar = [U[u_id] for u_id in utts_sofar_idx]
    # R[r_id].draw('tmp/rect_S1_utts.png', utts_sofar)

    # L1 = make_L1(lexicon)
    # l1_probs = L1(utts_sofar_idx)
    # # get argmax
    # rect_id_recovered = np.argmax(l1_probs)
    # rect_recovered = R[rect_id_recovered]
    # rect_recovered.draw('tmp/rect_S1_L1_recovered.png', utts_sofar)

    # rect_id_recovered1 = np.argmax(L1(top_utts_idx))
    # rect_recovered1 = R[rect_id_recovered1]
    # rect_recovered1.draw('tmp/rect_So_L1_recovered.png', top_utts)

    # data = get_S1_L1_data(lexicon, 2)
    # print (data[:10])

    # print (data[3])
    # u_idxs, rect_idxs = data[3]
    # R[rect_idxs[0]].draw('tmp/rect_data_S1_L1.png', [U[u] for u in u_idxs])

    # rect_global_rank = get_global_rect_id(lexicon, n_utters = 3, n_iters = 100000)
    # for r_id in [0,1,2,3,-1,-2,-3]:
    #     R[r_id].draw(f'tmp/rank{r_id}.png')