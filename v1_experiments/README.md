# here be some preliminary experiments
also some helpful utility functions you can steal

## small 1D line learning domain
refer to https://arxiv.org/abs/2007.05060 section 3 for problem statement

we consider a bigger problem on a grid-size of 1x8, so there are total of roughly 30 hypotheses.

we also build into the meaning matrix compound utterances composed of a pair of atomic utterances (wow this sentence reads like crap). i.e. of the form u1 and u2. see code for details

we wish to investigate if such compound utterances, when treated without ordering, can be pragmatically meaningful. and also we want to see if these compound utterances exhibits a global ranking

