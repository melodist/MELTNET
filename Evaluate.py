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
import ImageProcessing
from Extraction import PatchExtraction
from sklearn.cluster import KMeans
from datetime import datetime

path_model = './model/20191031_012618'
# Extract Features using trained network
# Load model
input_shape = (17 * 17)
embedding_size = 150
trained_model = NetworkKeras.create_base_network(input_shape, embedding_size)
trained_model.load_weights(path_model)

# Load Images
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]
path = './Examples'

# Make Results Folder
now = datetime.now()
path_result = f"./Results_{now.strftime('%Y%m%d_%H%M%S')}/"
os.makedirs(path_result)

# Print Patients Number
patient_dir = os.listdir(path)
print(f'Patients Number: {len(patient_dir)}')

for path_patient in patient_dir:
    addr_patient = f'{path}/{path_patient}/'
    path_files = path_result + path_patient + '/'
    os.makedirs(path_files)
    os.makedirs(f'{path_files}CT/')
    os.makedirs(f'{path_files}PT/')

    img_CT, img_PT = PatchExtraction.stackImages(addr_patient, ind_CT, ind_PT)
    patches_CT = PatchExtraction.patch_extraction(img_CT)
    patches_PT = PatchExtraction.patch_extraction(img_PT)

    # Extract Features
    print(f"Extract Features...")
    features = trained_model.predict([patches_CT, patches_PT], steps=1)

    # Using K-means
    print(f"K-Means Clustering...")
    num_labels = 10
    model_k_means = KMeans(n_clusters=num_labels)
    model_k_means.fit(features)

    # Merging Patches
    num_x = 44
    num_y = 30
    stride = 5

    label_predict = model_k_means.fit_predict(features)
    label_predict_batch = label_predict.reshape((-1, num_y * num_x))

    # Extract File Names
    for root, dirs, files in os.walk(os.path.join(path, path_patient)):
        file_list = files
        file_list.sort()

    print(f'Merging Patches...')
    for i, filename in enumerate(file_list):
        mask = ImageProcessing.merging_patches(label_predict_batch[i, :], num_labels, num_y, num_x, stride)
        for j in range(num_labels):
            ImageProcessing.save_image(mask[:, :, j], f'Results_{j}_' + filename, path_files)
        # save original image as reference
        cv2.imwrite(path_files + 'CT/' + filename, img_CT[i, :, :, 0]*255)
        cv2.imwrite(path_files + 'PT/' + filename, img_PT[i, :, :, 0]*255)

    ImageProcessing.ImageBlending(path_files, 10)
print(f"Done.")
