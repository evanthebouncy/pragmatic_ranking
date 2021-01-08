from prag_utils import *
import numpy as np
import argparse

L = 4

# make the hypothesis, utterances, and meaning_matrix for line game
def make_linegame():
    # generate the set of hypothesis : all possible contiguous segments in 1xL grid
    hypothesis = []
    for i in range(L):
        for j in range(L):
            if i <= j:
                hypothesis.append((i,j))
    # generate the set of atomic utterances : for each location loc, whether occupied or not
    atomic_utterances = []
    for loc in range(L):
        for occupied in [True,False]:
            atomic_utterances.append((loc,occupied))
    # generate the meaning matrix of atomic_utterances and hypothesis
    mm = []
    for loc,occupied in atomic_utterances:
        consistent_hypothesis_given_utter = []
        for i,j in hypothesis:
            if occupied == (i<=loc<=j):
                consistent_hypothesis_given_utter.append(1)
            else:
                consistent_hypothesis_given_utter.append(0)
        mm.append(consistent_hypothesis_given_utter)
    return hypothesis, atomic_utterances, np.array(mm)


