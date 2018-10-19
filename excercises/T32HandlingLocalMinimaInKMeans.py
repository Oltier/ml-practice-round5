import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import IntProgress

from T31Excercise import k_means
from excercises.T2Data import plotting


def empirical_risk(data, clusters, centroids):
    # INPUT: N x d data array, k x d array of k mean vectors (centroids),
    #       N x 1 array of cluster assignments.
    # OUTPUT: value of empirical risk

    ### STUDENT TASK ###
    # YOUR CODE HERE

    N = data.shape[0]

    _sum = 0

    for i in range(0, N):
        diff = data[i] - centroids[clusters[i]]
        norm = np.linalg.norm(diff, axis=0)
        _sum += norm

    risk = _sum / N

    return risk


def new_k_means(data, k, plot=True):
    # This will display a progress bar during k-mean execution
    f = IntProgress(description=f'KM (k={k}):', min=0, max=50)
    display(f)

    # initializing the array where we collect all cluster assignments
    cluster_collection = np.zeros((50, data.shape[0]), dtype=np.int32)
    # initializing the array where we collect all risk values
    risk_collection = np.zeros(50)

    for i in range(50):
        f.value += 1
        centroids, clusters = k_means(data, k, random_seed=i, plot=False)
        risk_collection[i] = empirical_risk(data, clusters, centroids)
        cluster_collection[i, :] = clusters

    # find the best cluster assignment and print the lowest found empirical risk
    min_ind = np.argmin(risk_collection)
    max_ind = np.argmax(risk_collection)
    if plot:
        print("Cluster division with lowest empirical risk")
        plotting(data, clusters=cluster_collection[min_ind, :])
        print("Cluster division with highest empirical risk")
        plotting(data, clusters=cluster_collection[max_ind, :])

        print('min empirical risk is ', np.min(risk_collection))

    # Let's remove progress bar
    f.close()
    return cluster_collection[min_ind, :], risk_collection


df = pd.read_csv("data.csv")
data = df.values
best_cluster, risk = new_k_means(data, 3)

risks = np.zeros(8)
for i in range(0, 8):
    best_cluster, risk = new_k_means(data, i + 1, plot=False)
    risks[i] = np.mean(risk)

fig = plt.figure(figsize=(8, 6))
plt.plot(range(1, 9), risks)
plt.xlabel('Number of clusters')
plt.ylabel('Empirical risk')
plt.title("The number of clusters vs the empirical risk")
plt.show()
