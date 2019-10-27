#!/usr/bin/env python
# coding: utf-8

# # Feature Extraction Network  
# Conv 50@5x5: Conv -> BN -> ReLU  
# Max Pooling 2x2  
# Conv 50@5x5  
# Max Pooling 2x2  
# FCN 1350  
# FCN 160  
# 
# Patch_CT : 17x17 stride 5  
# Patch_PT : 17x17 stride 5  
# 113 subjects  


import tensorflow as tf
import numpy as np
from Extraction import PatchExtraction
from Cluster import ClusterInitialization, ClusterMerging
import NetworkTensorflow
import NetworkKeras
import Cluster

tf.enable_eager_execution()

# Training Networks

# Extract Patches
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]
path = ''

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    img_CT, img_PT = PatchExtraction.stackImages(path, ind_CT, ind_PT)
    patches_CT = PatchExtraction.patch_extraction(img_CT)
    patches_PT = PatchExtraction.patch_extraction(img_PT)

    # Initialize Network
    input_shape = (17*17)
    embedding_size = 150
    base_network = NetworkKeras.create_base_network(input_shape, embedding_size)
    base_network.summary()

    # Extract Features using network
    features = base_network.predict([patches_CT, patches_PT])
    print(f"Shape of extracted features: {features.shape}")

    # Initialize clusters
    K_s = 20
    K_a = 5
    K_c_star = 100
    rand_samples = 1000

    rand_ind = np.random.choice(features.shape[0], rand_samples) # randomly choose rand_samples
    cluster = ClusterInitialization.Clusters(features[rand_ind], K_c_star, K_s, K_a)

    # sample number : patch number

