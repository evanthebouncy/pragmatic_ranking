import csv
import numpy as np

def process_data():
    cache = set()
    keep = []
    with open('20q_animal.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            row_id = str(row[1:])
            if row_id not in cache:
                cache.add(row_id)
                keep.append(row)
    return keep

def make_animalgame():
    interventions = []
    utterances = []
    hypotheses = []
    mm = []

    for i, row in enumerate(process_data()):
        # get the utterances
        if i == 0:
            interventions = row[1:]
            for inter in interventions:
                utterances.append(inter+'1')
                utterances.append(inter+'0')
        else:
            hypotheses.append(row[0])
            to_add = []
            for outcome in row[1:]:
                if outcome == '1':
                    to_add += [1,0]
                else:
                    to_add += [0,1]
            mm.append(to_add)
    return hypotheses, utterances, np.array(mm).transpose()



