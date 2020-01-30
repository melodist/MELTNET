""" ImageProcessing
    Image processing functions for evaluation
"""

import cv2
import numpy as np
from sklearn.metrics import euclidean_distances


def project_patches(labels, num_labels, num_y, num_x, stride, s_size):
    """ Project segmented images from labels of patches.
        This function will make empty image which has same size with original medical image.
        Empty image will be divided into subpatches which has square shape with s_size.
        Subpatches will have same labels with patches
        which has the nearest centroids from the centroids of each subpatch.

    Input
    ______
    labels: array for labels of patches
    num_labels: number of labels
    num_y: number of patches in y-axis
    num_x: number of patches in x-axis
    stride: stride of making pathces
    s_size: size of subpatches

    Output
    ______
    img_result: 3-D matrix for projected images. [num_labels, height, width]
    """

    img_result = np.zeros((num_y * stride, num_x * stride, num_labels))
    mesh = np.arange(num_y * num_x).reshape((num_y, num_x))
    s_half = int(s_size / 2)

    # Calculate centroids of subpatches
    xs_center = np.array(range(s_half, num_x * stride, s_size))
    ys_center = np.array(range(s_half, num_y * stride, s_size))

    # Calculate centroids of patches
    x_center = np.array(range(2, num_x * stride, stride))
    y_center = np.array(range(2, num_y * stride, stride))
    X, Y = np.meshgrid(x_center, y_center)
    centers = np.array([list(zip(x, y)) for x, y in zip(X, Y)])

    for x in xs_center:
        for y in ys_center:
            dists = euclidean_distances([[x, y]], centers.reshape(num_y * num_x, 2))

            # Reshape dists and find minimum index
            dists_reshape = dists.reshape(y_center.shape[0], x_center.shape[0])
            x_min = np.where(dists.min() == dists_reshape)[1][0]
            y_min = np.where(dists.min() == dists_reshape)[0][0]

            img_result[y - s_half:y + s_half + 1, x - s_half:x + s_half + 1, labels[mesh[y_min, x_min]]] += 1

    for i in range(num_labels):
        img_result[:, :, i] = img_result[:, :, i] / img_result[:, :, i].max() * 255

    return img_result


def save_image(image, filename, path):
    """ Save matrix as image file

    Input
    ______
    image: image data as matrix
    filename: file name to be saved
    path: path that file will be saved

    Output
    ______
    """
    fileaddr = path + filename
    cv2.imwrite(fileaddr, image)
