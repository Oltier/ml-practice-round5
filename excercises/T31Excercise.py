import numpy as np
import pandas as pd

from excercises.T2Data import plotting


def select_centroids(data, k, random_seed=1):
    # INPUT: N x d data array, k number of clusters.
    # OUTPUT: k x d array of k randomly assigned mean vectors with d dimensions.

    # Random seed will generate exactly same "random" values for each execution.
    # This will ensure similar results between students and avoid confusion.
    np.random.seed(seed=random_seed)

    centroids = np.zeros((k, data.shape[1]))
    for i in range(data.shape[1]):
        centroids[:, i] = np.random.uniform(np.min(data[:, i]),
                                            np.max(data[:, i]),
                                            size=(k))
    return centroids


def assign_points(data, centroids):
    # INPUT: N x d data array, k x d centroids array.
    # OUTPUT: N x 1 array of cluster assignments in {0,...,k-1}.

    N = data.shape[0]

    clusters = np.zeros(N, dtype=np.int32)

    # YOUR CODE HERE
    ### STUDENT TASK ###
    for i in range(0, data.shape[0]):
        diff = data[i] - centroids
        norm = np.linalg.norm(diff, axis=1)
        clusters[i] = np.argmin(norm, axis=0)

    return clusters


def move_centroids(data, old_centroids, clusters):
    # INPUT:  N x d data array, k x d centroids array, N x 1 array of cluster assignments
    # OUTPUT: k x d array of relocated centroids

    new_centroids = np.zeros(old_centroids.shape)
    ### STUDENT TASK ###
    # YOUR CODE HERE
    raise NotImplementedError()
    return new_centroids


df = pd.read_csv("data.csv")
data = df.values

# Step 3.1
centroids = select_centroids(data, 2)
# plotting(data, centroids)

# Step 3.2
clusters = assign_points(data, centroids)
plotting(data, centroids, clusters)


# Step 3.3
# new_centroids = move_centroids(data, centroids,clusters)
# plotting(data, new_centroids, clusters)
