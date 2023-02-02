from utils import *


if __name__ == '__main__':

    row_weights = [2, 5/3, 5/3, 8/3]
    col_weights = [11/10, 11/10, 63/40, 3/8]
    # make an outer product of the weights
    weights = np.outer(row_weights, col_weights)
    print (1 / weights)

    # lexicon = make_lexicon(20, 10, p=0.3)

    # scores = simulate(lexicon, population_number=1, n_games = 1000, simulation_n=1000)
    # # plot the scores
    # plt.plot(scores)
    # plt.show()


    # # run the rsa on the lexicon
    # lexicon, rsa_listeners, rsa_speakers = run_rsa(lexicon, n_runs=10)

    # rsa_s1 = RSA_Speaker(rsa_speakers[1])
    # rsa_l1 = RSA_Listener(rsa_listeners[1])

    # speaker = Speaker(lexicon)
    # listener = Listener(lexicon)
    
    # speaker_population = [speaker]
    # listener_population = [listener]

    # rsa_score = 0
    # for jj in range(1000):
    #     hypothesis_r = np.random.choice(range(lexicon.shape[1]))
    #     rsa_score += roundtrip(hypothesis_r, rsa_s1, rsa_l1)
    # print ("rsa score", rsa_score)

    # scores = []
    # # improve on each other for awhile
    # for _ in range(1000):
    #     speaker, score = speaker.improve(listener_population, n_games=1000)
    #     listener, score = listener.improve(speaker_population, n_games=1000)
    #     scores.append(score)

    # # plot the scores
    # plt.plot(scores)
    # plt.show()

    # print (scores[-1])