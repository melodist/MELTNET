"""Calculate Triplet loss for batch optimization
"""
from scipy.spatial import distance
import numpy as np
import tensorflow as tf
import ExecutionTime


def Label_to_table(labels):
    """ Convert label to table
    Input
    ______
    labels : [n_samples]
    Output
    ______
    table : [n_clusters, n_samples_in_each_cluster]

    1. Sort labels and argsort labels
    2. Append original index as a value of list
    3. If label changes, make a new list to append
    """

    labels_sorted = np.sort(labels)
    idx_sorted = np.argsort(labels)
    # print(labels_sorted, idx_sorted)

    n_cluster = 0
    label = labels_sorted[0]
    labels_from_one = np.zeros(labels.shape[0]).astype('uint8')

    # Change cluster label to 0,1,2...n_clusters
    for i in range(1, labels_sorted.shape[0]):
        if labels_sorted[i] != label:
            label = labels_sorted[i]
            n_cluster += 1
        labels_from_one[idx_sorted[i]] = n_cluster

    # Make table which has empty list as number of unique labels
    labels_tb = []
    for i in range(np.unique(labels).shape[0]):
        labels_tb.append([])

    for i in range(labels.shape[0]):
        labels_tb[labels_from_one[i]].append(i)

    return labels_tb


@ExecutionTime.execution_time
def organize_samples(X, y, num_neg_sampling):
    """Organize samples for triplet loss calculation

    Input
    ______
    X : Input features
    y : labels for input features

    Output
    ______
    {A, B, C} : [n_triplet, n_features]
    {A_ind, B_ind, C_ind}

    1. Gather index of samples in triplet
    2. use tf.gather_nd
    """
    # num_neg_sampling = 5

    y_table = Label_to_table(y)
    num_s = X.shape[0]
    print(f"num_s: {num_s}")
    n_clusters = len(y_table)
    if n_clusters == 1:
        return
    else:
        if n_clusters < num_neg_sampling:
            num_neg_sampling = n_clusters - 1

        num_triplet = 0
        for i in range(n_clusters):
            if len(y_table[i]) > 1:
                num_triplet += len(y_table[i]) * (len(y_table[i]) - 1) * num_neg_sampling / 2
        num_triplet = int(num_triplet)
        if num_triplet == 0:
            return

        A_ind = np.zeros(num_triplet).astype('uint16')
        B_ind = np.zeros(num_triplet).astype('uint16')
        C_ind = np.zeros(num_triplet).astype('uint16')
        id_triplet = 0

        for i in range(n_clusters):
            for m in range(len(y_table[i])):
                for n in range(m + 1, len(y_table[i])):
                    is_choosed = np.zeros(num_s)
                    while True:
                        id_s = np.random.randint(0, num_s)
                        id_t = y_table[i][m]
                        if is_choosed[id_s] == 0 and y[id_s] != y[id_t]:
                            A_ind[id_triplet] = y_table[i][m]
                            B_ind[id_triplet] = y_table[i][n]
                            C_ind[id_triplet] = id_s
                            is_choosed[id_s] = 1
                            id_triplet += 1
                        if is_choosed.sum() == num_neg_sampling:
                            break

        A = X[A_ind]
        B = X[B_ind]
        C = X[C_ind]

        return (A, B, C), (A_ind, B_ind, C_ind)


@ExecutionTime.execution_time
# Calculate Triplet loss for batch optimization v2
def Triplet_Calculate_for_batch_v2(triplets,
                                   lambda_tr=1.0, gamma_tr=2.0, margin_tr=0.5):
    """ Calculate Triplet loss for batch optimization

    v2. Use all samples in batch for x_i

    Input
    ______
    lambda_tr: weight to avoid
    gamma_tr: weight for affinities from same cluster
    margin_tr: margin threshold (alpha)
    triplets : matrix for triplet calculation (A, P, N)

    Output
    ______
    loss : value for triplet loss function

    loss = max(0, margin + p - n)

    p : positive
    n : negative

    1. Make the distance matrix for samples in batch
    2. Find x_k for each x_i
    3. Calculate triplet loss
    """
    a = triplets[0]
    p = triplets[1]
    n = triplets[2]
    num_triplets = triplets[0].shape[0]

    delta_pos = a - p
    delta_neg = a - n

    norm_delta_pos = tf.multiply(tf.norm(delta_pos), gamma_tr)
    norm_delta_neg = tf.norm(delta_neg)

    delta_pos_neg = tf.add(norm_delta_pos - norm_delta_neg, margin_tr)

    losses = tf.math.maximum(delta_pos_neg, 0)
    loss = tf.reduce_sum(losses) / num_triplets
    return loss


def Triplet_Calculate_for_batch_v1(c_a, K_c, batch_samples, batch_labels,
                                   lambda_tr=1.0, gamma_tr=2.0, margin_tr=0.5):
    """ Calculate Triplet loss for batch optimization

  Input
  ______
  c_a: index of most recently merged cluster
  K_c: integer constant
  lambda_tr: weight to avoid
  gamma_tr: weight for affinities from same cluster
  margin_tr: margin threshold (alpha)
  batch_samples: submatrix of samples used in batch-based learning
  batch_labels: submatrix of labels used in batch-based learning
  batch_index: index of samples used in batch-based learning
  w : weight matrix

  Output
  ______
  loss : value for triplet loss function

  loss = max(0, margin + p - n)

  p : positive
  n : negative

  1. Make the distance matrix for samples in batch
  2. Find x_k for each x_i
  3. Calculate triplet loss
  """

    # Make the distance matrix for samples in batch
    size_batch = batch_samples.shape[0]
    dist_batch = np.zeros((size_batch, size_batch))
    for i in range(size_batch):
        for j in range(size_batch):
            dist_batch[i, j] = distance.euclidean(batch_samples[i, :], batch_samples[j, :])

    # use anchor from cluster c_a
    anchors = np.where(batch_labels == c_a)[0]
    if anchors.size < 2:
        return 0

    loss = 0
    for anchor in anchors:
        # Select x_pos and x_neg / both are index of samples
        # x_pos : randomly selected from same cluster with anchor
        # x_neg : K_c neighbour samples of anchor
        set_pos = np.delete(anchors, np.where(anchors == anchor))
        x_pos = np.random.choice(set_pos, 1)[0]

        # Find x_k for each x_i
        set_neg = []
        n_neg = 0
        n = 0

        # Sort distance matrix to find K_c neighbours of anchor
        neigh_index = np.argsort(dist_batch[anchor, :])
        while n_neg < K_c and n < size_batch:
            ind_sorted = np.where(neigh_index == n)[0]
            if batch_labels[ind_sorted] != batch_labels[anchor] and n != anchor:
                set_neg.append(n)
                n_neg += 1
            n += 1

        # Calculate triplet loss
        for x_neg in set_neg:
            p = dist_batch[anchor, x_pos]
            n = dist_batch[anchor, x_neg]
            triplet_loss = np.max([0, gamma_tr * p - n + margin_tr])
            loss += - lambda_tr / (K_c - 1) * triplet_loss

    return loss
