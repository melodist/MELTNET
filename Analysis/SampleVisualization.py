# Sample Visualization
# 1-1. Extract Features using initial network
# 1-2. Extract Features using trained network
# 2. Using K-means to classify the patches
# 3. Dimension reduction using PCA
# 4. Visualize results

import tensorflow as tf
import NetworkKeras
import numpy as np
import cv2
import os
import time
from Extraction import PatchExtraction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime

tf.enable_eager_execution()

time_start = time.time()
path_model = './model/JULE/'
# Extract Features using trained network
# Load model
input_shape = (17 * 17)
embedding_size = 150
trained_model = NetworkKeras.create_base_network(input_shape, embedding_size, 0.5)
initial_model = NetworkKeras.create_base_network(input_shape, embedding_size, 0.5)
trained_model.load_weights(path_model)

# Load Images
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]
path = '../Test'

# Make Results Folder
now = datetime.now()
path_result = f"./Results_{now.strftime('%Y%m%d_%H%M%S')}/"
os.makedirs(path_result)

# Print Patients Number
patient_dir = os.listdir(path)
print(f'Patients Number: {len(patient_dir)}')

for path_patient in patient_dir:
    addr_patient = f'{path}/{path_patient}/'

    img_CT, img_PT = PatchExtraction.stackImages(addr_patient, ind_CT, ind_PT)
    patches_CT, patches_PT = PatchExtraction.patch_extraction_thres(img_CT, img_PT, 0)

    # Extract Features using initial network
    print(f"Extract Features using initial network...")
    features_init = initial_model.predict([patches_CT, patches_PT], steps=1)

    # Extract Features using trained network
    print(f"Extract Features...")
    features = trained_model.predict([patches_CT, patches_PT], steps=1)

    # Using K-means
    print(f"K-means Clustering...")
    num_labels = 5
    model_k_means = KMeans(n_clusters=num_labels, random_state=0)
    model_k_means.fit(features)
    label_predict = model_k_means.fit_predict(features)

    # Dimension reduction using PCA
    pca = PCA(n_components=2)
    features_low = pca.fit_transform(features)
    features_init_low = pca.transform(features_init)

    colors = ['salmon', 'orange', 'steelblue', 'violet', 'khaki']
    fig, ax = plt.subplots(2, figsize=(5, 5), constrained_layout=True)

    for i in range(5):
        data_init = features_init_low[label_predict == i]
        X_init = data_init[:, 0]
        Y_init = data_init[:, 1]
        ax[0].scatter(X_init, Y_init, color=colors[i], label=i, s=1)

        data = features_low[label_predict == i]
        X = data[:, 0]
        Y = data[:, 1]
        ax[1].scatter(X, Y, color=colors[i], label=i, s=1)

    ax[0].legend(loc='best')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].legend(loc='best')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    fig.suptitle('Distribution of patches')
    plt.savefig(f"{path_result}Plot_{path_patient}.png", format='png', dpi=300)

time_end = time.time()
print(f"Evaluation Finished! Elapsed time: {time_end - time_start}")