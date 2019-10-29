import cv2
import os
import numpy as np


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


# Visualize Results
def save_image(image, filename, path):
    fileaddr = path + filename
    cv2.imwrite(fileaddr, image)


def ImageBlending(diraddr):
    # Read Images
    file_list = os.listdir(diraddr)
    file_list.sort()

    for i in range(int(len(file_list)/3)):
        img_CT = cv2.imread(f'{diraddr}CT{i}.png')
        img_mask = 255 - cv2.imread(f'{diraddr}Results{i}.png')

        # Blend Images
        dst = cv2.addWeighted(img_CT, 0.3, img_mask, 0.7, 0)

        # Save Images
        cv2.imwrite(f'{diraddr}Overlay_{i}.png', dst)

    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('Image Blending Finished')

