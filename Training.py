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
from Extraction import PatchExtraction
import Cluster


# # Layers for Network

def convnn_CT(x):
    """Convolution Neural Networks for Feature Extraction using CT Images
    
    Input
    ___
    x : (n_samples, 17, 17)
    
    """
    # First Conv Layer: 17x17 to 15x15
    W_conv1 = weight_variable([3, 3, 1, 50])
    b_conv1 = bias_variable([50])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # First Pooling layer: 15x15 to 8x8.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Conv Layer: 8x8 to 6x6
    W_conv2 = weight_variable([3, 3, 50, 50])
    b_conv2 = bias_variable([50])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer: 6x6 to 3x3
    h_pool2 = max_pool_2x2(h_conv2)

    # Flatten the layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 50])

    return h_pool2_flat


def convnn_PT(x):
    """Convolution Neural Networks for Feature Extraction using PET Images
    
    Input
    ___
    x : (n_samples, 17, 17)
    
    """
    # First Conv Layer: 5x5 to 3x3
    W_conv1 = weight_variable([3, 3, 1, 50])
    b_conv1 = bias_variable([50])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # First Pooling layer: 15x15 to 8x8.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Conv Layer: 8x8 to 6x6
    W_conv2 = weight_variable([3, 3, 50, 50])
    b_conv2 = bias_variable([50])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer: 6x6 to 3x3
    h_pool2 = max_pool_2x2(h_conv2)

    # Flatten the layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 50])

    return h_pool2_flat


def conv2d(x, W):
    """conv2d는 full stride를 가진 2d convolution layer를 반환(return)한다."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2는 특징들(feature map)을 2X만큼 downsample한다."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable는 주어진 shape에 대한 weight variable을 생성한다."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable 주어진 shape에 대한 bias variable을 생성한다."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def fcn_dual(ct, pt):
    """Fully Connected Networks for Feature Extraction
    """
    # Merge two inputs into one vector
    x = tf.concat([ct, pt], 1)

    # FCN 1
    W_fc1 = weight_variable([3 * 3 * 50 * 2, 300])
    b_fc1 = bias_variable([450])
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # FCN 2
    W_fc2 = weight_variable([450, 150])
    b_fc2 = bias_variable([150])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    return h_fc2


# Training Networks
# Extract Patches
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]
path = ''

img_CT, img_PT = PatchExtraction.stackImages(path, ind_CT, ind_PT)
patches_CT = tf.image.extract_patches(img_CT, sizes=[1, 17, 17, 1],
                                      strides=[1, 5, 5, 1], padding='SAME')
patches_PT = tf.image.extract_patches(img_PT, sizes=[1, 17, 17, 1],
                                      strides=[1, 5, 5, 1], padding='SAME')
print(f"Patch Extraction Completed: {patches_CT.shape}")

# Initialize clusters
# sample number : patch number

