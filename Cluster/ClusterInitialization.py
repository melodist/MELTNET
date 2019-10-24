# Cluster Initialization
"""Initialize clusters and label

"""
import numpy as np
import tensorflow as tf
import ExecutionTime
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance


class Clusters:
    def __init__(self, samples, n_c_star, **kwargs):
        print(f"Initialize Clusters...")
        self.samples = samples
        self.n_c_star = n_c_star

        if "K_s" in kwargs:
            self.K_s = kwargs["K_s"]
        else:
            self.K_s = 5

        if "K_c" in kwargs:
            self.K_c = kwargs["K_c"]
        else:
            self.K_c = 5

        if "K_0" in kwargs:
            self.K_0 = kwargs["K_0"]
        else:
            self.K_0 = 1

        if "a" in kwargs:
            self.a = kwargs["a"]
        else:
            self.a = 1

        self.labels = cluster_initialize(self.samples, self.K_0)
        self.n_clusters = np.unique(self.labels).shape[0]
        print(f"Number of Initial Clusters: {self.n_clusters}")
        print(f"Calculate weights for all samples...")
        self.w = calculate_weights(self.samples, self.K_s, self.a)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     aff_table = self.aff_initialize_GPU()
        #     self.aff_table = sess.run(aff_table)
        self.aff_table = self.aff_initialize_CPU()
        self.K = self.aff_knn()
        print(f"Cluster Initialization Completed!")

    # Initialize Affinity Table
    @ExecutionTime.execution_time
    def aff_initialize_GPU(self):
        """Initialize the affintiy table and make the K^c-nearest cluster set

      Input
      ______
      w : Weight matrix of samples
      labels : cluster index of samples

      Output
      ______
      aff_table : [n_clusters, n_clusters]
      """
        with tf.device('/GPU:0'):
            aff_table = np.zeros((self.n_clusters, self.n_clusters))
            print(f"Calculate affinity tables for all samples...")

            print(f"Calculation Numbers: {self.n_clusters * (self.n_clusters + 1) / 2}")
            # Make affinity table
            for i in range(self.n_clusters):
                for j in range(i + 1, self.n_clusters):  # A[i, j] = A[j, i]
                    aff_table[i, j] = self.aff_between_two_clusters_GPU(i, j).eval()
                    aff_table[j, i] = aff_table[i, j]

        print(f"Affinity Table Initialization Completed!")
        return aff_table

    @ExecutionTime.execution_time
    def aff_initialize_CPU(self):
        """Initialize the affintiy table and make the K^c-nearest cluster set

      Input
      ______
      w : Weight matrix of samples
      labels : cluster index of samples

      Output
      ______
      aff_table : [n_clusters, n_clusters]
    """

        aff_table = np.zeros((self.n_clusters, self.n_clusters))
        print(f"Calculate affinity tables for all samples...")

        print(f"Calculation Numbers: {self.n_clusters**2/2}")
        # Make affinity table
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):  # A[i, j] = A[j, i]
                aff_table[i, j] = self.aff_between_two_clusters_CPU(i, j)
                aff_table[j, i] = aff_table[i, j]  # copy values for optimization

        print(f"Affinity Table Initialization Completed!")
        return aff_table

    def aff_knn(self):
        """ Make K^c-nearest cluster set
        """

        K = np.argsort(-self.aff_table[0, :])[:self.K_c]  # argsort sorts values for descending order
        for i in range(1, self.n_clusters):
            K_row = np.argsort(-self.aff_table[i, :])[:self.K_c]
            K = np.vstack([K, K_row])  # vstack consider 1-d array as (1,n) 2-d array
        print(f"KN-N Table Initialization Completed!")
        return K

    # Merge two clusters to new cluster
    def merge_two_clusters(self, c_a, c_b):
        """
      Merge two clusters and update cluster labels.
      label > c_b : label = label - 1

      Input
      ______
      c_a : small cluster number
      c_a : large cluster number
      labels : cluster labels

      Output
      ______
      labels : updated cluster labels
      """
        # Merge two clusters
        self.labels[self.labels == c_b] = c_a
        # Update the label
        self.labels[self.labels > c_b] = self.labels[self.labels > c_b] - 1

    # Find the set which has maximum affinity
    def aff_find_maximum(self):
        """ Find the set which has maximum affinity

      Input
      ______

      Output
      ______
      c1, c2 : cluster label
      """
        aff_max = np.max(self.aff_table)
        aff_max_ind = np.where(self.aff_table == aff_max)

        return aff_max_ind[0][0], aff_max_ind[1][0]

    def aff_between_two_clusters_GPU(self, c_a, c_b):
        """ Calculate Affinity between two clusters

        :param c_a: index of cluster a
        :param c_b: index of cluster b
        :return: affinity between two clusters
        """
        # Find sample index which belongs to each cluster using list comprehension
        ind_c_a = np.where(self.labels == c_a)[0]
        ind_c_b = np.where(self.labels == c_b)[0]

        # Make submatrix of w
        w_a_b = self.w[ind_c_a][:, ind_c_b]
        w_b_a = w_a_b.T

        size_c_a = len(ind_c_a)
        size_c_b = len(ind_c_b)
        if size_c_a == 0:
            print(f"Zero at c_a: {c_a}")
        if size_c_b == 0:
            print(f"Zero at c_a: {c_b}")

        ones_a = tf.ones((size_c_a, 1), dtype=tf.float32)
        ones_b = tf.ones((size_c_b, 1), dtype=tf.float32)
        ones_a_T = tf.ones((1, size_c_a), dtype=tf.float32)
        ones_b_T = tf.ones((1, size_c_b), dtype=tf.float32)

        x_a_b = tf.constant(w_a_b, dtype=tf.float32)
        x_b_a = tf.constant(w_b_a, dtype=tf.float32)

        # Calculate Affinity between two clusters
        A = (1 / size_c_a ** 2 * tf.matmul(tf.matmul(tf.matmul(ones_a_T, x_a_b), x_b_a), ones_a) +
             1 / size_c_b ** 2 * tf.matmul(tf.matmul(tf.matmul(ones_b_T, x_b_a), x_a_b), ones_b))

        return tf.squeeze(A)

    def aff_between_two_clusters_CPU(self, c_a, c_b):
        """ Calculate Affinity between two clusters

        :param c_a: index of cluster a
        :param c_b: index of cluster b
        :return: affinity between two clusters
        """
        # Find sample index which belongs to each cluster using list comprehension
        ind_c_a = np.where(self.labels == c_a)[0]
        ind_c_b = np.where(self.labels == c_b)[0]

        # Make submatrix of w
        w_a_b = self.w[ind_c_a][:, ind_c_b]
        w_b_a = w_a_b.T

        size_c_a = len(ind_c_a)
        size_c_b = len(ind_c_b)
        if size_c_a == 0:
            print(f"Zero at c_a: {c_a}")
        if size_c_b == 0:
            print(f"Zero at c_a: {c_b}")

        ones_a = np.ones((size_c_a, 1))
        ones_b = np.ones((size_c_b, 1))

        # Calculate Affinity between two clusters
        A = (1 / size_c_a ** 2 * np.matmul(np.matmul(np.matmul(ones_a.T, w_a_b), w_b_a), ones_a) +
             1 / size_c_b ** 2 * np.matmul(np.matmul(np.matmul(ones_b.T, w_b_a), w_a_b), ones_b))

        return A.squeeze()

    def aff_between_two_samples(self, ind_a, ind_b):
        """

        :param ind_a: Index of sample a
        :param ind_b: Index of sample b
        :return: Affinity between two samples
        """
        # Make submatrix of w
        w_a_b = self.w[ind_a][ind_b]
        w_b_a = self.w[ind_b][ind_a]

        # Calculate Affinity between two clusters
        A = 2 * w_a_b * w_b_a

        return A

    # Update Affinity table
    def aff_update(self, c_a, c_b):
        """Update affinity table using AGDL

      Input
      ______
      w : Weight matrix of samples
      labels_old : cluster index of samples
      aff_table : table of affinities between clusters
      K : K_c-nearest neighbor cluster sets
      c_a, c_b : labels of merged clusters

      Output
      ______
      aff_table : [n_clusters, n_clusters]
      K : [n_clusters, K_c]


      1. Update affinity for the clusters who has c_a or c_b as neighbor
      2. Update labels and K
      3. Remove c_b from aff_table
      """
        K_c = self.K.shape[1]

        # Update Labels
        self.merge_two_clusters(c_a, c_b)

        # Find index of clusters who includes c_a or c_b as a neighbor
        list_c_a = np.where(self.K == c_a)[0]
        list_c_b = np.where(self.K == c_b)[0]
        list_union = np.union1d(list_c_a, list_c_b)
        list_to_update = np.setdiff1d(list_union, np.array([c_a, c_b]))
        list_to_update[list_to_update > c_b] = list_to_update[list_to_update > c_b] - 1
        # print(f"list to update: {list_to_update}")

        # Update affinity
        for c in list_to_update:
            self.aff_table[c, c_a] = self.aff_between_two_clusters_CPU(c_a, c)
            self.aff_table[c_a, c] = self.aff_table[c, c_a]

        # Update K
        self.K[c_a, :] = [x for x in np.argsort(-self.aff_table[c_a, :])[1:K_c + 1]]
        # print(f"K[c_a, :]: {K[c_a, :]}")
        K = np.delete(self.K, c_b, 0)
        K[K > c_b] = K[K > c_b] - 1

        # print(f"K.shape: {K.shape}")
        # print(f"labels.max: {labels.max()}\n")

        # Remove c_b from table
        self.aff_table = np.delete(self.aff_table, c_b, 0)
        self.aff_table = np.delete(self.aff_table, c_b, 1)

    # Merging clusters loop
    @ExecutionTime.execution_time
    def aff_cluster_loop(self, eta=0.9):
        """ Loop for merging clusters

      Input
      ______
      eta :

      Output
      ______
      labels : [n_samples]
      aff_table :
      K

      1. Calculate the loop number
      L1-1. Find the clusters who has maximum affinity
      L1-2. Merging the clusters and update the affinity table
      L2. Update the CNN 20 times
      """

        # Calculate the loop number
        loops = int(np.unique(self.labels).shape[0] * eta)
        print(f'Affinity cluster loop size: {loops}')
        print(f'Affinity cluster loop start!')
        for i in range(loops):
            c_a, c_b = self.aff_find_maximum()
            # print(c_a, c_b)
            self.aff_update(c_a, c_b)

    def is_finished(self):
        """ Check number of cluster reaches number of desired clusters

        :return: boolean
        """
        return self.n_clusters > self.n_c_star


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
    # print(cluster_init)

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
