import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import IntProgress
from scipy.stats import multivariate_normal

cmpd = ['orangered', 'dodgerblue', 'springgreen']
cmpcent = ['red', 'darkblue', 'limegreen']


def initialize_parameters(data, k, random_seed=1):
    # Random seed will generate exactly same "random" values for each execution.
    # This will ensure similar results between students and avoid confusion.
    np.random.seed(seed=random_seed)

    means = np.zeros((k, data.shape[1]))
    for i in range(data.shape[1]):
        means[:, i] = np.random.uniform(np.min(data[:, i]),
                                        np.max(data[:, i]),
                                        size=(k))
    covariances = np.zeros((k, data.shape[1], data.shape[1]))

    for f in range(k):
        covariances[f] = np.identity(data.shape[1])
    coefficients = np.ones(k) * 0.5
    return means, covariances, coefficients


def plot_GMM(data, means, covariances, k, cluster_vectors=None):
    if cluster_vectors is None:
        plt.scatter(data[:, 0], data[:, 1], s=13, alpha=0.5)
    else:
        clusters = np.argmax(cluster_vectors, axis=0)
        plt.scatter(data[:, 0], data[:, 1], c=[cmpd[i] for i in clusters], s=13, alpha=0.5)

    # Visualization of results
    x_plot = np.linspace(19, 35, 100)
    y_plot = np.linspace(0, 12, 100)

    for i in range(k):
        x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
        z = plt.mlab.bivariate_normal(x_mesh, y_mesh, np.sqrt(covariances[i, 0, 0]),
                                      np.sqrt(covariances[i, 1, 1]), means[i, 0], means[i, 1], covariances[i, 0, 1])
        plt.contour(x_mesh, y_mesh, z, 4, colors=cmpcent[i], alpha=0.5)
        plt.scatter([means[i, 0]], [means[i, 1]], marker='x', c=cmpcent[i])

    plt.title("Soft clustering with GMM")
    plt.xlabel("feature x_1: customers' age")
    plt.ylabel("feature x_2: money spent during visit")
    plt.show()


def update_degrees_of_belonging(data, means, covariances, coefficients, k):
    cluster_vectors = np.zeros((k, data.shape[0]))
    for i in range(data.shape[0]):
        belonging_all = np.sum(
            [coefficients[f] * multivariate_normal.pdf(data[i], means[f], covariances[f]) for f in range(k)])
        for t in range(k):
            cluster_vectors[t, i] = coefficients[t] * multivariate_normal.pdf(data[i], means[t],
                                                                              covariances[t]) / belonging_all

    return cluster_vectors


def update_GMM_pars(data, cluster_vectors, k):
    means_new = np.zeros((k, data.shape[1]))
    covariances_new = np.zeros((k, data.shape[1], data.shape[1]))
    coefficients_new = np.zeros(k)

    for i in range(k):
        sum_k = np.sum(cluster_vectors[i])
        coefficients_new[i] = sum_k / len(data)
        means_new[i] = data.T @ cluster_vectors[i] / sum_k
        for l in range(data.shape[0]):
            covariances_new[i] += np.outer(data[l] - means_new[i], data[l] - means_new[i]) * cluster_vectors[i, l]
        covariances_new[i] = covariances_new[i] / sum_k
    return means_new, covariances_new, coefficients_new


def GMM_clustering(data, k, num_iters, random_seed=0):
    # Step 1:
    means, covariances, coefficients = initialize_parameters(data, k, random_seed)

    # This will display a progress bar during GMM execution
    f = IntProgress(description=f'GMM (k={k}):', min=0, max=num_iters)
    display(f)
    for i in range(num_iters):
        # Step 2:
        cluster_vectors = update_degrees_of_belonging(data, means, covariances, coefficients, k)
        # Step 3:
        means, covariances, coefficients = update_GMM_pars(data, cluster_vectors, k)

        # Iterate progress bar
        f.value += 1

    return means, covariances, cluster_vectors


df = pd.read_csv("data.csv")
data = df.as_matrix()

# Step 1
means, covariances, coefficients = initialize_parameters(data, 3)
plot_GMM(data, means, covariances, 3)

# Step 2
# cluster_vectors = update_degrees_of_belonging(data, means, covariances, coefficients, 3)
# plot_GMM(data, means, covariances, 3, cluster_vectors)


# Step 3
# means, covariances, coefficients = update_GMM_pars(data, cluster_vectors, 3)
# plot_GMM(data, means, covariances, 3, cluster_vectors)

# Step 4
# means, covariances, cluster_vectors = GMM_clustering(data, 3, 50)
# plot_GMM(data, means, covariances, 3, cluster_vectors)
# print("The means are", means)
# print("The covariance matrices are", covariances)
