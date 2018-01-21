import pandas
import numpy
import sys
sys.path.insert(0, "../src")

from clustering import kmeans


if __name__ == '__main__':

    data = pandas.read_csv('../data/bezdekIris.data', header=None)
    examples = data.iloc[:,0:4].values
    #labels = data.iloc[:,4].values
    nb_clusters = 3
    seed = 20061982
    actual = kmeans(examples, nb_clusters, seed)
    expected = numpy.loadtxt('kmeans.txt')
    if numpy.array_equal(actual, expected):
        print('SUCCESS')
    else:
        print("FAILED")
    #nrow = data.shape[0]
    #ntrain = int(0.7 * nrow)
    #ntest = nrow - ntrain

    #random.sample(range(0, nrow), ntest)

