import cv2
import os
import numpy as np
from sklearn.metrics import euclidean_distances


# Merging Patches
def merging_patches(labels, num_labels, num_y, num_x, stride):
    mask_image = np.zeros((num_y * stride, num_x * stride, num_labels))
    mesh = np.arange(num_y * num_x).reshape((num_y, num_x))
    for x in range(num_x):
        for y in range(num_y):
            mask_image[stride * y:stride * y + 16, stride * x:stride * x + 16, labels[mesh[y, x]]] += 1

    for i in range(num_labels):
        mask_image[:, :, i] = mask_image[:, :, i] / mask_image[:, :, i].max() * 255

    return mask_image


# Project Patches
def project_patches(labels, num_labels, num_y, num_x, stride):
    mask_image = np.zeros((num_y * stride, num_x * stride, num_labels))
    mesh = np.arange(num_y * num_x).reshape((num_y, num_x))

    # Calculate centroids of patches
    x_center = np.array(range(2, 220, 5))
    y_center = np.array(range(2, 150, 5))
    X, Y = np.meshgrid(x_center, y_center)
    centers = np.array([list(zip(x, y)) for x, y in zip(X, Y)])

    for x in x_center:
        for y in y_center:
            dists = euclidean_distances([[x, y]], centers.reshape(30 * 44, 2))

            # Reshape dists and find minimum index
            dists_reshape = dists.reshape(30, 44)
            x_min = np.where(dists.min() == dists_reshape)[0][0]
            y_min = np.where(dists.min() == dists_reshape)[1][0]

            mask_image[y-2:y+2, x-2:x+2, labels[mesh[y_min, x_min]]] += 1

    for i in range(num_labels):
        mask_image[:, :, i] = mask_image[:, :, i] / mask_image[:, :, i].max() * 255

    return mask_image


# Visualize Results
def save_image(image, filename, path):
    fileaddr = path + filename
    cv2.imwrite(fileaddr, image)


def ImageBlending(diraddr, feature_num):
    # Read Images
    ct_list = os.listdir(f'{diraddr}CT/')
    ct_list.sort()

    for i in ct_list:
        for j in range(feature_num):
            img_CT = cv2.imread(f'{diraddr}CT/{i}')
            img_mask = 255 - cv2.imread(f'{diraddr}Results_{j}_{i}')
            # Make mask violet
            img_mask[:, :, 1:2] = 0

            # Blend Images
            dst = cv2.addWeighted(img_CT, 0.5, img_mask, 0.5, 0)

            # Save Images
            cv2.imwrite(f'{diraddr}Overlay_{j}_{i}.png', dst)

    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('Image Blending Finished')

