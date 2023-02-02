from utils import *
from exp_exists_orders import get_order_listener, get_order_speaker, check_order_consistent
# import counter 
from collections import Counter

# track orders in order to see if they're stable
def track_stable_orders(listeners):
    orders_chain = [get_order_listener(listener) for listener in listeners]
    stable_formation = dict()
    for i in range(len(orders_chain)):
        cur_ords = orders_chain[i]
        rest_ords = orders_chain[i+1:]
        # union rest_ords together
        rest_together = set()
        for ord in rest_ords:
            rest_together = rest_together.union(ord)
        for ord in cur_ords:
            # check the reverse order in together or not
            ord_rev = (ord[1], ord[0])
            if ord_rev not in rest_together:
                if ord not in stable_formation:
                    stable_formation[ord] = i
    
    return stable_formation

# try to find a lexicon that is as unstable as possible
# shouldn't worry now after we found the bug
def debug_find_unstable_lexicon():
    max_unstable = 0
    most_unstable_lexicon = None
    for i in range(1000):
        lexicon = make_lexicon(5, 5)
        lexicon, listeners, speakers = run_rsa(lexicon, n_runs=100)
        stable_formation = track_stable_orders(listeners)
        stable_values = stable_formation.values()
        
        # for the case of purly symmetric lexicon
        if len(stable_values) == 0:
            continue
        # check the max
        if max(stable_values) > max_unstable:
            max_unstable = max(stable_values)
            most_unstable_lexicon = lexicon
            print (max_unstable)
            print (stable_formation)
            visualize_lexicon(lexicon, save_path='drawings/unstable_lexicon.png', show_numbers=True)
    return most_unstable_lexicon

def get_stable_orders(listeners):
    stable_formation = track_stable_orders(listeners)
    stable_values = stable_formation.values()
    total_orders = len(stable_values)
    # count the occurances in stable_values
    stable_counter = Counter(stable_values)
    for key in stable_counter:
        stable_counter[key] = stable_counter[key] / total_orders
    return stable_counter

def get_formation_for_lexicon_size(lexicon_size, prob):
    formation = dict()
    N_SAMPLE = 20
    for _ in range(N_SAMPLE):
        # make a random lexicon
        lexicon = make_lexicon(lexicon_size, lexicon_size, prob)
        # run rsa
        lexicon, listeners, speakers = run_rsa(lexicon, n_runs=100)
        xx = get_stable_orders(listeners)
        for key in xx:
            if key not in formation:
                formation[key] = []
            formation[key].append(xx[key])
    return formation

def exp_get_formation(size_low_end, size_high_end, prob=0.5):
    plot_1_points = []
    data_rows = []
    end_range = 0
    for lexicon_size in range(size_low_end, size_high_end):
        print (lexicon_size)
        # fractions of formation at L1, contains many samples (100)
        formation1 = get_formation_for_lexicon_size(lexicon_size, prob)[1]
        plot_1_points.append(formation1)
        for n_sample, formation_frac in enumerate(formation1):
            data_rows.append([f'Ptrue={prob}', lexicon_size, n_sample, formation_frac])
        end_range = size_low_end + lexicon_size - 1
        # this is convergence (i.e. all orderings are formed in L1)
        if end_range > 20:
            if np.mean(formation1) > 0.999:
                break
    return data_rows

if __name__ == '__main__':
    import csv
    csv_header = ['Ptrue', 'lexicon_size', 'n_sample', 'formation_frac']
    csv_body = []
    for prob in [0.5, 0.2, 0.1]:
        data = exp_get_formation(2, 100, prob)
        csv_body += data
    with open('formation.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_body)
