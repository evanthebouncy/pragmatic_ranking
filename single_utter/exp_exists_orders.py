from utils import *
# import the class of Counter
from collections import Counter

def get_order_listener(listener):
    m, n = listener.shape
    orders = set()
    for row_id in range(m):
        row = listener[row_id]
        # grab all the non-zero indexes of this row
        non_zero_indexes = np.nonzero(row)[0]
        for idx1 in non_zero_indexes:
            for idx2 in non_zero_indexes:
                val1, val2 = row[idx1], row[idx2]
                # get the absolute difference between the two values
                diff = abs(val1 - val2)
                # if the difference is less than 1e-6, then they are the same due to floating point error
                if diff < 1e-6:
                    continue
                if val1 > val2:
                    orders.add((idx1, idx2))
                if val2 > val1:
                    orders.add((idx2, idx1))
    return orders

def get_order_speaker(speaker):
    # simply transpose the speaker and pretend it is a listener
    return get_order_listener(speaker.transpose())

def check_order_consistent(orders):
    for order in orders:
        ord_rev = (order[1], order[0])
        if ord_rev in orders:
            return False
    return True

# this should always work, since we have the proof
def exp_check_order_exists(n_runs = 10000):
    listner_ordered, speaker_ordered = 0, 0
    listener_checked, speaker_checked = 0, 0

    for i in range(n_runs):
        m = np.random.randint(10, 21)
        n = np.random.randint(10, 21)    
        lexicon = make_lexicon(m, n)
        lexicon, listeners, speakers = run_rsa(lexicon)

        for listener in listeners:
            listener_checked += 1
            l_ords = get_order_listener(listener)
            if check_order_consistent(l_ords):
                listner_ordered += 1
        print ('Listener ordered : %d / %d' % (listner_ordered, listener_checked))

        for speaker in speakers[1:]:
            speaker_checked += 1
            s_ords = get_order_speaker(speaker)
            if check_order_consistent(s_ords):
                speaker_ordered += 1
        print ('Speaker ordered : %d / %d' % (speaker_ordered, speaker_checked))

def diagnose_inversion(listener, listener_next):
    m, n = listener.shape
    l_ords = get_order_listener(listener)
    l_next_ords = get_order_listener(listener_next)
    for ord in l_ords:
        ord_rev = (ord[1], ord[0])
        if ord_rev in l_next_ords:
            print ('Inversion found!')
            print (ord, ord_rev)
            print (listener)
            print (listener_next)

# this should fail, giving evidence that, once an ordering is established
# it is possible to reverse it in later stages of the rsa chain
def exp_check_order_inversion(n_runs = 1000):

    for i in range(n_runs):
        m, n = 5, 5
        lexicon = make_lexicon(m, n)
        lexicon, listeners, speakers = run_rsa(lexicon)

        l_all_ords = [get_order_listener(listener) for listener in listeners]

        for l_id in range(len(listeners)-1):
            l_ords = l_all_ords[l_id]
            l_next_ords = l_all_ords[l_id+1]
            # union together the two
            together = l_ords.union(l_next_ords)
            # check consistency
            if not check_order_consistent(together):
                print ('Listener %d and %d are not consistent' % (l_id, l_id+1))
                visualize_lexicon(lexicon, 'drawings/order_violating_lexicon.png')
                diagnose_inversion(listeners[l_id], listeners[l_id+1])
                visualize_lexicon(listeners[l_id], f'drawings/order_violating_listener_{l_id}.png', True)
                visualize_lexicon(listeners[l_id+1], f'drawings/order_violating_listener_{l_id+1}.png', True)
                for l_idd in range(len(listeners)):
                    visualize_lexicon(listeners[l_idd], f'drawings/order_violating_l_{l_idd}.png', True)
                assert 0


if __name__ == '__main__':
    exp_check_order_exists(1000)
    exp_check_order_inversion(100)