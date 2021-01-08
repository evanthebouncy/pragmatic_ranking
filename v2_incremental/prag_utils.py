import numpy as np
import matplotlib.pyplot as plt


# row and colomn normalizations
def normalise(mat, axis):
    if axis == 0:
        row_sums = mat.sum(axis=1)
        new_matrix = mat / row_sums[:, np.newaxis]
        return new_matrix
    if axis == 1:
        col_sums = mat.sum(axis=0)
        new_matrix = mat / col_sums[np.newaxis, :]
        return new_matrix

# use these and don't fudge with the axis lol
def make_speaker(mat):
    return normalise(mat, 1)

def make_listener(mat):
    return normalise(mat, 0)

# sharpen the distribution using soft-max, essentially make the rich richer
def sharpen_speaker(mat):
    return make_speaker(np.e**(100*mat))

def sharpen_listener(mat):
    return make_listener(np.e**(100*mat))

# accuracy between S and L can be computed as follows
# compute P(w'= w) = integrate_u Pspeak(u | w) Plisten(w' | u)
def comm_acc(S,L):
    w_to_w = (S*L).sum(axis=0)
    return w_to_w.mean()

# visualize a matrix (either meaning matrix, or S, or L)
def draw(x, name):
    plt.figure(dpi=400)
    plt.imshow(x, cmap='gray')
    plt.savefig(f"drawings/{name}.png")
    plt.close()


if __name__ == '__main__':
    # take a meaning matrix, flip it a few times, communication accuracy should go up
    M   = np.array([ [1,1,1],
                     [0,1,1],
                     [0,0,1],
                  ])
    draw(M,"M")
    L = make_listener(M)

    for i in range(20):
        S = normalise(L, 1)
        L = normalise(S, 0)
        end_acc = comm_acc(S, L)
        print (end_acc)
    draw(S,"S")
    draw(L,"L")
    print (S,"\n", L)

