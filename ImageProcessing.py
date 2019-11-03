import cv2
import os
import numpy as np
from sklearn.metrics import euclidean_distances


# Merging Patches
def merging_patches(labels, num_labels, num_y, num_x, stride):
    mask_image = np.zeros((num_y * stride, num_x * stride, num_labels))
    mesh = np.arange(num_y * num_x).reshape((num_y, num_x))
    print(f'Merging Patches...')
    for x in range(num_x):
        for y in range(num_y):
            mask_image[stride * y:stride * y + 17, stride * x:stride * x + 17, labels[mesh[y, x]]] += 1

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
            x_min = np.where(dists.min() == dists_reshape)[1][0]
            y_min = np.where(dists.min() == dists_reshape)[0][0]

            mask_image[y-2:y+3, x-2:x+3, labels[mesh[y_min, x_min]]] += 1

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
            img_result = cv2.imread(f'{diraddr}Results_{j}_{i}')
            img_mask_plane = make_mask(img_CT, img_result)

            # Make mask violet
            img_mask = np.repeat(img_mask_plane[:, :, np.newaxis], 3, axis=2)
            img_mask[:, :, 1:2] = 0

            # Blend Images
            dst = cv2.addWeighted(img_CT, 0.7, img_mask, 0.3, 0)

            # Save Images
            cv2.imwrite(f'{diraddr}Overlay_{j}_{i}.png', dst)

    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('Image Blending Finished')


def make_mask(img_CT, img_result):
    gray = 255 - cv2.cvtColor(img_CT, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Finding Contour
    images, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_contours.append(area)

    area_contours = np.array(area_contours)
    sort_contours = np.argsort(area_contours)

    contour_lung = [contours[sort_contours[-1]], contours[sort_contours[-2]]]
    mask = np.zeros(img_CT.shape, np.uint8)
    cv2.drawContours(mask, contour_lung, -1, 255, thickness=cv2.FILLED)

    # Bitwise calculation
    img_result_plane = cv2.bitwise_not(img_result[:, :, 0])
    mask_plane = mask[:, :, 0]
    result2 = cv2.bitwise_and(img_result_plane, mask_plane)

    kernel_close = np.ones((5, 5), np.uint8)
    result3 = cv2.morphologyEx(255 - result2, cv2.MORPH_CLOSE, kernel_close)

    return 255 - result3
