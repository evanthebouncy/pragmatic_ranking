import numpy as np 

# some awful global variables indicating the max size of the world
MAX_LEN = 6

# a rectangle is parameterized by Top/Bottom/Left/Right

class Rect:

    # initialize a rectangle with top, bottom, left, right
    def __init__(self, T,B,L,R) -> None:
        self.T = T
        self.B = B
        self.L = L
        self.R = R

    def __str__(self) -> str:
        return "R({},{},{},{})".format(self.T, self.B, self.L, self.R)

    def __repr__(self) -> str:
        return "R({},{},{},{})".format(self.T, self.B, self.L, self.R)

    # check if a point is inside the rectangle
    def is_inside(self, x, y) -> bool:
        return self.T <= y <= self.B and self.L <= x <= self.R

    # turn it into a function that returns a boolean using the is_inside function
    def __call__(self, x, y) -> bool:
        return self.is_inside(x, y)

    # given a list of (x,y),bool pairs check if the rectangle is consistent with the points
    def consistent(self, point_bools: list) -> bool:
        for (x,y),b in point_bools:
            if not self(x,y) == b:
                return False
        return True

    # generate a png image of the rectangle
    def draw(self, filename: str, examples = []) -> None:
        import matplotlib.pyplot as plt
        # set the boundaries of the plot to be 10 by 10
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
        # draw the rectangle with thick borders
        # draw the top line from L,T to R,T
        plt.plot([self.L, self.R], [self.T, self.T], 'k-', linewidth=2)
        # draw the bottom line from L,B to R,B
        plt.plot([self.L, self.R], [self.B, self.B], 'k-', linewidth=2)
        # draw the left line from L,T to L,B
        plt.plot([self.L, self.L], [self.T, self.B], 'k-', linewidth=2)
        # draw the right line from R,T to R,B
        plt.plot([self.R, self.R], [self.T, self.B], 'k-', linewidth=2)

        # draw the examples
        for i, ex in enumerate(examples):
            # if the ex does not contain a boolean
            if type(ex[1]) != bool:
                if self.is_inside(ex[0], ex[1]):
                    # draw a big green dot
                    plt.plot(ex[0], ex[1], 'go', markersize=10)
                else:
                    plt.plot(ex[0], ex[1], 'ro', markersize=10)
            else:
                xy, b = ex
                if b:
                    plt.plot(xy[0], xy[1], 'go', markersize=10)
                else:
                    plt.plot(xy[0], xy[1], 'ro', markersize=10)
                text_color = 'green' if b else 'red'
                text_offset = 0.2
                plt.text(xy[0]+text_offset, xy[1]+text_offset, str(i), color=text_color)

        plt.savefig(filename)
        plt.close()

def rect_is_valid(T,B,L,R) -> bool:
    non_empty = T < B and L < R
    not_big = (B-T) + (R-L) <= MAX_LEN
    # make sure they are on the canvas
    not_outside = T >= 0 and B <= MAX_LEN and L >= 0 and R <= MAX_LEN
    return non_empty and not_big and not_outside

# make all the hypothesis
def enum_all_rects():
    all_rects = []
    for T in range(0, MAX_LEN+1):
        for B in range(T+1, MAX_LEN+1):
            for L in range(0, MAX_LEN+1):
                for R in range(L+1, MAX_LEN+1):
                    if rect_is_valid(T,B,L,R):
                        all_rects.append(Rect(T,B,L,R))
    return all_rects
    
# make all the utterances
def enum_all_utts():
    all_examples = []
    for i in range(0, MAX_LEN+1):
        for j in range(0, MAX_LEN+1):
            for bool in [True, False]:
                all_examples.append(((i,j),bool))
    return all_examples

def get_lexicon():
    rects = enum_all_rects()
    utts = enum_all_utts()
    lexicon = np.zeros((len(utts), len(rects)))
    for u_id, u in enumerate(utts):
        for r_id, r in enumerate(rects):
            lexicon[u_id, r_id] = r.consistent([u])
    return rects, utts, lexicon


if __name__ == '__main__':
    rect = Rect(1,6,1,6)
    print(rect.is_inside(5,5))
    print(rect.is_inside(6,6))
    print(rect.is_inside(7,7))
    print(Rect(1,6,1,6)(5,6))

    Rect(1,3,4,9).draw('tmp/rect.png', [(1,1), (2,2), (3,3), (4,4), (4,1), (5,2), (9,3)])

    test_coords = []
    for x in range(0, MAX_LEN+1):
        for y in range(0, MAX_LEN+1):
            test_coords.append((x,y))
    examples = [((x,y),Rect(1,3,4,9)(x,y)) for x,y in test_coords]
    Rect(1,3,4,9).draw('tmp/rect2.png', examples)

    all_rects = enum_all_rects()
    all_examples = enum_all_utts()
    print(len(all_rects))
    print(len(all_examples))
    R, U, lexicon = get_lexicon()
    print(lexicon.shape)
    print (lexicon)

    all_rects[34].draw('tmp/rect3.png')