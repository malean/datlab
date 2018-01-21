import numpy
import random

   
def kmeans(examples, nb_clusters, seed = None):

    random.seed(seed)

    # random initialization of group centers
    random_indices = random.sample(range(0, examples.shape[0]), nb_clusters)
    centers = examples[random_indices]
    
    # memory allocation 
    members = [[] for i in range(nb_clusters)]
    distances = numpy.ndarray(nb_clusters)
    centers_old = numpy.ndarray(nb_clusters)
    
    while ( not numpy.array_equal(centers, centers_old) ):
        
        # assign center for each sample
        for i, example in enumerate(examples):
            for j, center in enumerate(centers):
                distances[j] = numpy.linalg.norm(example - center)
            members[numpy.argmin(distances)].append(i)

        # update centers
        centers_old = centers
        for i in range(nb_clusters):
            centers[i] = numpy.mean(examples[members[i]])

    return centers