import random

# take a corpus of utterances, a listener response model, a set of hypothesis
# attempt to find an ordering of hypothesis that is consistent
def find_ordering(utt_corpus, L1, H):
    ords = [i for i in range(len(H))]
    ords_to_match = []
    for s1_us in utt_corpus:
        l1 = L1(s1_us)
        l1_zip = [(i,ll1) for i,ll1 in enumerate(l1)]
        l1_ord = sorted(l1_zip, key = lambda x: -x[1])
        l1_ord_filt = [x[0] for x in l1_ord if x[1] != float('-inf')]
        ords_to_match.append(l1_ord_filt)

    def swap(i,j,l):
        a,b = l[i],l[j]
        l[i],l[j] = b,a


    for _ in range(10000):
        random_row = random.choice(ords_to_match)
        if len(random_row) < 2:
            continue
        random_pair = random.sample(random_row, 2)
        pair_ord = random_row.index(random_pair[0]) < random_row.index(random_pair[1])
        pair_rank_ord = ords.index(random_pair[0]) < ords.index(random_pair[1])
        if pair_ord != pair_rank_ord:
            # print (f"swapped {H[random_row.index(random_pair[0])]} {H[random_row.index(random_pair[1])]}")
            swap(ords.index(random_pair[0]),ords.index(random_pair[1]),ords)
    return ords

def make_Lo(ords, H, L0):
    def Lo(us):
        l0 = L0(us)
        ret = []
        for h_id in ords:
            if l0[h_id] > 0:
                ret.append(h_id)
        return ret
    return Lo