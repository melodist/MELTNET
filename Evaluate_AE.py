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
import time
import ImageProcessing
from Extraction import PatchExtraction
from sklearn.cluster import KMeans
from datetime import datetime

tf.enable_eager_execution()

time_start = time.time()
path_model = './model/AE_5patients/'
# Extract Features using trained network
# Load model
input_shape = (17 * 17)
embedding_size = 150

trained_model_CT = NetworkKeras.create_autoencoder(input_shape)
trained_model_CT.load_weights(path_model + 'CT/')
trained_model_CT.summary()

trained_model_PT = NetworkKeras.create_autoencoder(input_shape)
trained_model_PT.load_weights(path_model + 'PT/')
trained_model_PT.summary()

# Make feature extraction model

feature_extractor_CT = tf.keras.models.Model(inputs=trained_model_CT.input,
                                             outputs=trained_model_CT.get_layer('tf_op_layer_l2_normalize').output)
feature_extractor_PT = tf.keras.models.Model(inputs=trained_model_PT.input,
                                             outputs=trained_model_PT.get_layer('tf_op_layer_l2_normalize_2').output)

# Load Images
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]
path = './Test'

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
    os.makedirs(f'{path_files}Overlay/')
    os.makedirs(f'{path_files}Features/')

    img_CT, img_PT = PatchExtraction.stackImages(addr_patient, ind_CT, ind_PT)
    patches_CT, patches_PT = PatchExtraction.patch_extraction_thres(img_CT, img_PT, 0)

    # Extract Features
    print(f"Extract Features...")
    features_CT = feature_extractor_CT.predict(patches_CT, steps=1)
    features_PT = feature_extractor_PT.predict(patches_PT, steps=1)
    features = np.hstack((features_CT, features_PT))

    # Using K-means
    print(f"K-means Clustering...")
    num_labels = 5
    model_k_means = KMeans(n_clusters=num_labels, random_state=0)
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
        mask = ImageProcessing.project_patches(label_predict_batch[i, :], num_labels, num_y, num_x, stride)
        for j in range(num_labels):
            ImageProcessing.save_image(mask[:, :, j], f'./Features/Features_{j}_' + filename, path_files)
        # save original image as reference
        cv2.imwrite(path_files + 'CT/' + filename, img_CT[i, :, :, 0])
        cv2.imwrite(path_files + 'PT/' + filename, img_PT[i, :, :, 0])

    # print(f'Blending Images...')
    # ImageProcessing.ImageBlending(path_files, num_labels)

time_end = time.time()
print(f"Evaluation Finished! Elapsed time: {time_end - time_start:.2f}")
