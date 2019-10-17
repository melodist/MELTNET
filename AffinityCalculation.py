"""# Functions for Affinity
1. Initialize Affinity Table
2. Calculate KN-N for affinity table
1. Affinity between two clusters
2. Affinity between two samples
"""
import numpy as np
from ExecutionTime import execution_time

# Initialize Affinity Table
@execution_time
def Aff_initialize(w, labels):
    """Initialize the affintiy table and make the K^c-nearest cluster set

  Input
  ______
  w : Weight matrix of samples
  labels : cluster index of samples

  Output
  ______
  aff_table : [n_clusters, n_clusters]
  """
    n_clusters = np.unique(labels).shape[0]
    aff_table = np.zeros((n_clusters, n_clusters))

    # Make affinity table
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):  # A[i, j] = A[j, i]
            aff_table[i, j] = Aff_between_two_clusters(i, j, w, labels)
            aff_table[j, i] = aff_table[i, j]  # copy values for optimization

    return aff_table


def Aff_knn(aff_table, K_c):
    """ Make K^c-nearest cluster set
  """
    n_clusters = aff_table.shape[0]
    K = np.argsort(-aff_table[0, :])[:K_c]  # argsort sorts values for descending order
    for i in range(1, n_clusters):
        K_row = np.argsort(-aff_table[i, :])[:K_c]
        K = np.vstack([K, K_row])  # vstack consider 1-d array as (1,n) 2-d array

    return K


# Update Affinity table
def Aff_update(w, labels_old, aff_table, K, c_a, c_b):
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
    K_c = K.shape[1]

    # Update Labels
    labels = Merge_two_clusters(c_a, c_b, labels_old)

    # Find index of clusters who includes c_a or c_b as a neighbor
    list_c_a = np.where(K == c_a)[0]
    list_c_b = np.where(K == c_b)[0]
    list_union = np.union1d(list_c_a, list_c_b)
    list_to_update = np.setdiff1d(list_union, np.array([c_a, c_b]))
    list_to_update[list_to_update > c_b] = list_to_update[list_to_update > c_b] - 1
    # print(f"list to update: {list_to_update}")

    # Update affinity
    for c in list_to_update:
        aff_table[c, c_a] = Aff_between_two_clusters(c_a, c, w, labels)
        aff_table[c_a, c] = aff_table[c, c_a]

    # Update K
    K[c_a, :] = [x for x in np.argsort(-aff_table[c_a, :])[1:K_c + 1]]
    # print(f"K[c_a, :]: {K[c_a, :]}")
    K = np.delete(K, c_b, 0)
    K[K > c_b] = K[K > c_b] - 1

    # print(f"K.shape: {K.shape}")
    # print(f"labels.max: {labels.max()}\n")

    # Remove c_b from table
    aff_table1 = np.delete(aff_table, c_b, 0)
    aff_table2 = np.delete(aff_table1, c_b, 1)

    return labels, aff_table2, K


# Find the set which has maximum affinity
def Aff_find_maximum(aff_table):
    """ Find the set which has maximum affinity

  Input
  ______
  aff_table : table of affinities between clusters

  Output
  ______
  c1, c2 : cluster label
  """
    aff_max = np.max(aff_table)
    aff_max_ind = np.where(aff_table == aff_max)

    return aff_max_ind[0][0], aff_max_ind[1][0]


def Aff_between_two_clusters(c_a, c_b, w, labels):
    """ Calculate Affinity between two clusters

    :param c_a: index of cluster a
    :param c_b: index of cluster b
    :param w: weight matrix
    :param labels: label matrix
    :return: affinity between two clusters
    """
    # Find sample index which belongs to each cluster using list comprehension
    ind_c_a = np.where(labels == c_a)[0]
    ind_c_b = np.where(labels == c_b)[0]

    # Make submatrix of w
    w_a_b = w[ind_c_a][:, ind_c_b]
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


def Aff_between_two_samples(ind_a, ind_b, w):
    """

    :param ind_a: Index of sample a
    :param ind_b: Index of sample b
    :param w: weight table
    :return: Affinity between two samples
    """
    # Make submatrix of w
    w_a_b = w[ind_a][ind_b]
    w_b_a = w[ind_b][ind_a]

    # Calculate Affinity between two clusters
    A = 2 * w_a_b * w_b_a

    return A


# Merge two clusters to new cluster
def Merge_two_clusters(c_a, c_b, labels):
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
    labels[labels == c_b] = c_a
    # Update the label
    labels[labels > c_b] = labels[labels > c_b] - 1

    return labels


# Merging clusters loop
@execution_time
def Aff_cluster_loop(w, labels, aff_table, K, eta=0.9):
    """ Loop for merging clusters

  Input
  ______
  w :
  labels :
  aff_table :
  eta :
  K :

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
    K_c = K.shape[1]

    # Calculate the loop number
    loops = int(np.unique(labels).shape[0] * eta)
    for i in range(loops):
        c_a, c_b = Aff_find_maximum(aff_table)
        # print(c_a, c_b)
        labels, aff_table, K = Aff_update(w, labels, aff_table, K, c_a, c_b)

    return labels, aff_table, K