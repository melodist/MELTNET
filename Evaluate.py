# Evaluate
# 1. Extract Features using trained network
# 2. Using K-means to classify the patches
# 3. Merging Patches
# 4. Visualize results

import tensorflow as tf
import numpy as np
import NetworkKeras
from Extraction import PatchExtraction
from sklearn.cluster import k_means

path = './'
# Extract Features using trained network
# Load model
input_shape = (17 * 17)
embedding_size = 150
trained_model = NetworkKeras.create_base_network(input_shape, embedding_size)
trained_model.load_weights(path)

# Load Images
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]
path = './Example'

img_CT, img_PT = PatchExtraction.stackImages(path, ind_CT, ind_PT)
patches_CT = PatchExtraction.patch_extraction(img_CT)
patches_PT = PatchExtraction.patch_extraction(img_PT)

# Extract Features
features = trained_model.predict([patches_CT, patches_PT])

# Using K-means
model_k_means = k_means(n_clusters=2)
model_k_means.fit(features)

label_predict = model_k_means.fit_predict(features)

# Merging Patches
# 1. Make Empty mask with 512 * 512
# 2.
#
def merging_patches():