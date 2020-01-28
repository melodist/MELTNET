# Sample Visualization
# 1-1. Extract Features using initial network
# 1-2. Extract Features using trained network
# 2. Using K-means to classify the patches
# 3. Dimension reduction using PCA
# 4. Visualize results

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine.training import Model

import NetworkKeras
import os
import time
from Extraction import PatchExtraction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime

tf.enable_eager_execution()

time_start = time.time()
path_model = './model/AE/'

# Extract Features using trained network
# Load model
input_shape = (17 * 17)

initial_model_CT = NetworkKeras.create_autoencoder(input_shape)
initial_model_PT = NetworkKeras.create_autoencoder(input_shape)

trained_model_CT = NetworkKeras.create_autoencoder(input_shape)
trained_model_CT.load_weights(path_model + 'CT')

trained_model_PT = NetworkKeras.create_autoencoder(input_shape)
trained_model_PT.load_weights(path_model + 'PT')

# Make feature extraction model
initial_extractor_CT = tf.keras.models.Model(inputs=initial_model_CT.input,
                                             outputs=initial_model_CT.get_layer('tf_op_layer_l2_normalize').output)
initial_extractor_PT = tf.keras.models.Model(inputs=initial_model_PT.input,
                                             outputs=initial_model_PT.get_layer('tf_op_layer_l2_normalize_2').output)

feature_extractor_CT = tf.keras.models.Model(inputs=trained_model_CT.input,
                                             outputs=trained_model_CT.get_layer('tf_op_layer_l2_normalize_4').output)
feature_extractor_PT = tf.keras.models.Model(inputs=trained_model_PT.input,
                                             outputs=trained_model_PT.get_layer('tf_op_layer_l2_normalize_6').output)

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
    addr_patient = f'{path}/{path_patient}/'\

    img_CT, img_PT = PatchExtraction.stackImages(addr_patient, ind_CT, ind_PT)
    patches_CT, patches_PT = PatchExtraction.patch_extraction_thres(img_CT, img_PT, 0)

    # Extract Features using initial network
    print(f"Extract Features using initial network...")
    features_init_CT = initial_extractor_CT.predict(patches_CT, steps=1)
    features_init_PT = initial_extractor_PT.predict(patches_PT, steps=1)
    features_init = np.hstack((features_init_CT, features_init_PT))

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
