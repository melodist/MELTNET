# Evaluate
# 1. Extract Features using trained network
# 2. Using K-means to classify the patches
# 3. Merging Patches
# 4. Visualize results

import tensorflow as tf
import numpy as np
import NetworkKeras
import cv2
import os
from Extraction import PatchExtraction
from sklearn.cluster import KMeans
from datetime import datetime


# Merging Patches
def merging_patches(labels, num_y, num_x, stride):
    mask_image = np.zeros((num_y * stride, num_x * stride))
    mesh = np.arange(num_y * num_x).reshape((num_y, num_x))
    for x in range(num_x):
        for y in range(num_y):
            mask_image[stride * y:stride * y + 16, stride * x:stride * x + 16] += labels[mesh[y, x]]
    mask_image = mask_image / mask_image.max() * 255

    return mask_image


# Visualize Results
def save_image(image, filename, path):
    fileaddr = path + filename
    cv2.imwrite(fileaddr, image)


path_model = './model/'
# Extract Features using trained network
# Load model
input_shape = (17 * 17)
embedding_size = 150
trained_model = NetworkKeras.create_base_network(input_shape, embedding_size)
trained_model.load_weights(path_model)

# Load Images
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]
path = './Example'

img_CT, img_PT = PatchExtraction.stackImages(path, ind_CT, ind_PT)
patches_CT = PatchExtraction.patch_extraction(img_CT)
patches_PT = PatchExtraction.patch_extraction(img_PT)

# Extract Features
print(f"Extract Features...")
features = trained_model.predict([patches_CT, patches_PT], steps=1)

# Using K-means
print(f"K-Means Clustering...")
model_k_means = KMeans(n_clusters=2)
model_k_means.fit(features)

# Merging Patches
num_x = 44
num_y = 30
stride = 5

now = datetime.now()
path_files = f"./Results_{now}/"
os.makedirs(path_files)
label_predict = model_k_means.fit_predict(features)
label_predict_batch = label_predict.reshape((-1, num_y * num_x))

for i in range(label_predict_batch.shape[0]):
    filename = str(i) + '.png'
    mask = merging_patches(label_predict_batch[i, :], num_y, num_x, stride)
    save_image(mask, 'Results' + filename, path_files)
    # save original image as reference
    cv2.imwrite(path_files + 'CT' + filename, img_CT[i, :, :, 0]*255)
    cv2.imwrite(path_files + 'PT' + filename, img_PT[i, :, :, 0]*255)

print(f"Done.")
