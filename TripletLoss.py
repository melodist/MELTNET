"""Calculate Triplet loss for batch optimization
"""
from scipy.spatial import distance
import numpy as np


def Triplet_Calculate_for_batch(c_a, K_c, batch_samples, batch_labels,
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
