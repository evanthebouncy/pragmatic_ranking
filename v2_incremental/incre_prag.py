from prag_utils import *
from line1 import *
from animals import make_animalgame

from prag_ordering import *
import random

def make_L0(M, H_prior):
    # the function L0 takes in multiple utterances [u1 u2 u3]
    # and return a distribution over hypothesis H
    def L0(us):
        ret = []
        for h,h_prior in enumerate(H_prior):
            h_sat = 1 if (0 not in [M[u][h] for u in us]) else 0
            ret.append(h_sat * h_prior)
        ret = np.array(ret)
        if sum(ret) == 0:
            return ret
        return ret / sum(ret)
    return L0

def make_S1(L0, U_prior, U):
    # the function S1 takes in multiple utterances [u1 u2 u3]
    # and also takes in a hypothesus h
    # and computes the conditional probability S1([u1 u2 u3] | h)
    # this is given as a sequence:
    # P_S(u1 | h), P_S(u2 | h u1), P_S(u3 | h u1 u2)
    def S1(us, h):
        prob = 1
        utts_sofar = []

        ret = []
        for u in us:
            # compute all alternatives
            alt_terms = []
            for uu in range(len(U)):
                uu_prior = U_prior[uu]
                u_together = utts_sofar + [uu]
                pl0 = L0(u_together)[h]
                term = pl0 * uu_prior
                alt_terms.append(term)
            top_term = alt_terms[u]
            bot_term = sum(alt_terms)
            utts_sofar.append(u)

            ret.append((top_term, bot_term))

        return ret
    return S1

def make_L1(L0,S1,H):

    def L1(us):
        l0 = L0(us)
        ret = []
        for h in range(len(H)):
            if l0[h] > 0:
                ps1 = S1(us,h)
                term_top = sum([np.log(x[0]) for x in ps1])
                term_bot = sum([np.log(x[1]) for x in ps1])
                ret.append(term_top - term_bot)
            else:
                ret.append(float('-inf'))
        return ret

    return L1

def make_S1_sampler(L0, U_prior, U):
    # take in a h and a number k, sample best utterance of size k
    def S1(h, k):
        prob = 1
        utts_sofar = []

        ret = []
        for _ in range(k):
            # compute all alternatives
            alt_terms = []
            for uu in range(len(U)):
                uu_prior = U_prior[uu]
                u_together = utts_sofar + [uu]
                pl0 = L0(u_together)[h]
                term = pl0 * uu_prior
                alt_terms.append(term)
            best_u = np.argmax(alt_terms)
            utts_sofar.append(best_u)
        return utts_sofar
    return S1

# === tests === #
def make_rand_game():
    H = [i for i in range(200)]
    U = [j for j in range(10)]
    M = []
    for u in U:
        to_add = [1 if random.random() > 0.5 else 0 for h in H]
        M.append(to_add)

    def check_identifiability(mm):
        # todo: make sure every hypothesis can be identified
        pass
    return H, U, M

def test1():
    H, U, M = make_linegame()
    H_prior = [1/len(H) for x in H]
    U_prior = [1/len(U) for u in U]
    
    L0 = make_L0(M, H_prior)
    print(L0([1,7]))
    S1 = make_S1(L0, U_prior, U)
    S1_sample = make_S1_sampler(L0, U_prior, U)
    L1 = make_L1(L0,S1,H)

    for k in [1,2]:
        for h in H:
            s1_us = S1_sample(H.index(h),k)
            s1_uss = [U[uu] for uu in s1_us]
            print (h, s1_uss, H[np.argmax(L1(s1_us))])

if __name__ == '__main__':
    H, U, M = make_linegame()
    H, U, M = make_rand_game()
    H, U, M = make_animalgame()

    print (H)
    print (U)
    print (M.shape)

    H_prior = [1/len(H) for x in H]
    U_prior = [1/len(U) for u in U]
    
    L0 = make_L0(M, H_prior)
    S1 = make_S1(L0, U_prior, U)
    S1_sample = make_S1_sampler(L0, U_prior, U)
    L1 = make_L1(L0,S1,H)

    def go_until_good(h):
        for k in [1,2,3,4,5,6,7,8,9,10]:
            s1_us = S1_sample(H.index(h),k)
            l1 = L1(s1_us)
            l1_zip = [(i,ll1) for i,ll1 in enumerate(l1)]
            l1_ord = sorted(l1_zip, key = lambda x: -x[1])
            l1_ord_filt = [x[0] for x in l1_ord if x[1] != float('-inf')]
            if l1_ord_filt[0] == H.index(h):
                return s1_us, l1_ord_filt
        return s1_us, l1_ord_filt

    s1_corpus = []
    comm_complexity = []
    for h in H:
        utts, ords = go_until_good(h)
        s1_corpus.append(utts)
        print (f'{h} --S1-> {[U[u] for u in utts]} --L1-> {[H[h] for h in ords]}')
        comm_complexity.append(len(utts))
    print (sum(comm_complexity) / len(H))

    ords = find_ordering(s1_corpus, L1, H)
    print ("pragmatic orderings . . .")
    print ([H[o] for o in ords])

    print ("=== comparing L1 with Lo ===")
    Lo = make_Lo(ords, H, L0)
    for h in H:
        utts, ords = go_until_good(h)
        print (f'{h} --S1-> {[U[u] for u in utts]} --L1-> {[H[h] for h in ords]}')
        print (f'{h} --S1-> {[U[u] for u in utts]} --Lo-> {[H[h] for h in Lo(utts)]}')
        

