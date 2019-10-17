# Cluster Initialization
"""Initialize clusters and label

"""
import numpy as np
import ExecutionTime
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance


class Clusters:
    def __init__(self, samples, **kwargs):
        print(f"Initialize Clusters...")
        self.samples = samples
        self.n_clusters = -1

        if "K_s" in kwargs:
            self.K_s = kwargs["K_s"]
        else:
            self.K_s = 5

        if "K_0" in kwargs:
            self.K_0 = kwargs["K_0"]
        else:
            self.K_0 = 1

        if "a" in kwargs:
            self.a = kwargs["a"]
        else:
            self.a = 1

        self.labels = cluster_initialize(self.samples, self.K_0)
        print(f"Calculate weights for all samples...")
        self.w = calculate_weights(self.samples, self.K_s, self.a)

        print(f"Cluster Initialization Completed!")

@ExecutionTime.execution_time
def cluster_initialize(samples, K_0):
    """

    :param samples:
    :param K_0:
    :return:
    """
    # Finds the K-neighbors of a point
    clf = NearestNeighbors(n_neighbors=K_0)
    clf.fit(samples)
    _, cluster_init = clf.kneighbors()
    print(cluster_init)

    labels = np.full(samples.shape[0], -1)
    ind_cluster = 0

    # Define the label using Depth-First Search
    for i in range(samples.shape[0]):
        if labels[i] == -1:
            labels = dfs(labels, cluster_init, i, ind_cluster)
            ind_cluster += 1

    return labels


def dfs(labels, graph, start, index):
    visited = []
    stack = [start]

    while stack:
        n = stack.pop()
        if n not in visited and labels[n] == -1:
            visited.append(n)
            labels[n] = index
            stack += set(graph[n]) - set(visited)

    # print(f"visited: {visited}, i = {index}")
    return labels


@ExecutionTime.execution_time
def calculate_weights(samples, K_s, a):
    """
    Generate Weight Matrix of samples
    Input
    ------
    samples : [n_samples, n_features]
    knn_ind : [n_samples]

    Output
    ------
    w : [n_samples, n_samples]
    w[i, j] = w_ij
    """
    n_samples = samples.shape[0]
    w = np.zeros((n_samples, n_samples))

    # Calculate sigma_square
    clf_aff = NearestNeighbors(n_neighbors=K_s)
    clf_aff.fit(samples)
    knn_dist, knn_ind = clf_aff.kneighbors()
    dsts = np.sum(knn_dist)
    sigma_square = a / (n_samples * K_s) * dsts
    print(f"Total distance: {dsts:.2f}")  # using f-string formatting
    print(f"sigma_square: {sigma_square:.2f}")  # It requires python >= 3.6

    # Calculate weights
    for i in range(n_samples):
        for j in knn_ind[i]:
            dst = distance.euclidean(samples[i, :], samples[j, :])
            w[i, j] = np.exp(-dst ** 2 / sigma_square)

    return w
