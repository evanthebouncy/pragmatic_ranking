import numpy as np
import matplotlib.pyplot as plt

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

# take a lexicon, make respective listeners and speakers using RSA
def run_rsa(lexicon, n_runs = 100):
    # set rtol and atol for np.allclose when running too many iterations
    rtol, atol = 1e-3, 1e-3
    listeners = []
    speakers = [lexicon]
    # converged = False
    for i in range(n_runs):
        # if converged:
        #     listeners.append(listeners[-1])
        #     speakers.append(speakers[-1])
        last_speaker = speakers[-1]
        # make listener by normalize all rows
        listener = last_speaker / np.sum(last_speaker, axis=1, keepdims=True)
        listeners.append(listener)
        # make speaker by normalize all columns
        speaker = listener / np.sum(listener, axis=0, keepdims=True)
        speakers.append(speaker)

        # we need to check for convergence in the last 2 listeners and speakers
        # if they are the same, we can stop to prevent errors from compounding
        if i >= 1:
            if np.allclose(listeners[-1], listeners[-2], rtol, atol) or np.allclose(speakers[-1], speakers[-2], rtol, atol):
                listeners[-1] = listeners[-2]
                speakers[-1] = speakers[-2]
                break
    # change the first speaker to S0
    speakers[0] = lexicon / np.sum(lexicon, axis=0, keepdims=True)
    return lexicon, listeners, speakers

# get the round-trip efficiency of a speaker and a listener
def get_roundtrip_prob(speaker, listener):
    round_trip = np.matmul(speaker.T, listener)
    # take the avg of the diagonal
    return np.mean(np.diag(round_trip))

# visualize a lexicon using imshow
def visualize_lexicon(lexicon, save_path = None, show_numbers = False):
    # note : reverse 1 and 0 for imshow
    plt.imshow(1 - lexicon, cmap='gray')
    if show_numbers:
        for i in range(lexicon.shape[0]):
            for j in range(lexicon.shape[1]):
                numbr = lexicon[i, j]
                # round to 3 decimal places
                numbr = round(numbr, 3)
                plt.text(j, i, numbr, ha="center", va="center", color="b")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    # clear and close plt
    plt.clf()
    plt.close()

if __name__ == '__main__':
    lexicon = make_lexicon(10, 5)
    lexicon, listeners, speakers = run_rsa(lexicon)
    s1 = speakers[1]
    l0 = listeners[0]
    l1 = listeners[1]
    # print(s1, l1)
    # # multiply s1 and l0 using matrix multiplication
    print (get_roundtrip_prob(s1, l0))
    print (get_roundtrip_prob(s1, l1))

    # make sure my understanding of matrix multiplication is correct lol
    # upper left corner should be 0.68, meaning w1 transmit to w1' with prob 0.68
    ss = np.array([[0.1, 0.8], [0.2, 0.1], [0.7, 0.1]])
    ll = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    print(np.matmul(ss.T, ll))
    # the round-trip acc should be around 0.6 to 0.8
    print(get_roundtrip_prob(ss, ll))

    visualize_lexicon(lexicon)
    visualize_lexicon(listeners[0], show_numbers = True)
    visualize_lexicon(speakers[1], show_numbers = True)
    visualize_lexicon(speakers[2], show_numbers = True)

    # ===== paper example =====
    lexicon_paper = np.array([[1,0,0],[1,1,0],[0,1,1]])
    _, listeners, speakers = run_rsa(lexicon_paper)
    s0 = speakers[0]
    s1 = speakers[1]
    l0 = listeners[0]
    l1 = listeners[1]
    # print(s1, l1)
    # # multiply s1 and l0 using matrix multiplication
    print (get_roundtrip_prob(s0, l0))
    print (get_roundtrip_prob(s1, l0))
    print (get_roundtrip_prob(s1, l1))
    print (get_roundtrip_prob(s1, np.array([[1,0,0],[0,1,0],[0,0,1]])))