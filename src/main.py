import pandas
import random

data = pandas.read_csv('../data/bezdekIris.data')

nrow = data.shape[0]
ntrain = int(0.7 * nrow)
ntest = nrow - ntrain

random.sample(range(0, nrow), ntest)


