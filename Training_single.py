#!/usr/bin/env python
# coding: utf-8

# JULE using CT image only

import tensorflow as tf
import numpy as np
import TripletLossAdaptedFromTF
import os
import NetworkKeras
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from Extraction import PatchExtraction
from Cluster import ClusterInitialization


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch.
    inv: return base_lr * (1 + gamma * iter) ^ (- power)
    """
    gamma = 0.0001
    power = 0.75
    return lr * (1 + gamma * epoch) ** (-power)


time_start = time.time()
tf.enable_eager_execution()

# Training Networks

# Extract Patches
ind_CT = [[230, 380], [150, 370]]
path = './Training'
patient_list = os.listdir(path)
print(f'Number of patients: {len(patient_list)}')

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():

# Initialize Base Network
input_shape = (17 * 17)
embedding_size = 150
base_network = NetworkKeras.create_single_network(input_shape, embedding_size)
# base_network.summary()

# Initialize Entire Network
buffer_size = 10000
batch_size_per_replica = 128

input_CT = tf.keras.Input(shape=(17 * 17), name='Input_CT')
input_PT = tf.keras.Input(shape=(17 * 17), name='Input_PT')
input_labels = tf.keras.Input(shape=(1,), name='Input_label')

embeddings = base_network(input_CT)
labels_plus_embeddings = tf.keras.layers.concatenate([embeddings, input_labels])
print(labels_plus_embeddings)
model_triplet = tf.keras.Model(inputs=[input_CT, input_labels],
                               outputs=labels_plus_embeddings)

# model_triplet.summary()
tf.keras.utils.plot_model(model_triplet, to_file='model_triplet_single.png',
                          show_shapes=True, show_layer_names=True)
callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
opt = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=5e-5)
model_triplet.compile(loss=TripletLossAdaptedFromTF.triplet_loss_adapted_from_tf,
                      optimizer=opt)

# Make log file
now = datetime.now()
dir_model = f"./model/{now.strftime('%Y%m%d_%H%M%S')}/"
os.makedirs(dir_model)
f = open(f"{dir_model}log.txt", "w")
num_exp = 0
thres = 80

for patient in patient_list:
    # Make Patches
    img_CT, img_PT = PatchExtraction.stackImages(f'{path}/{patient}/', ind_CT, ind_CT)
    patches_CT, patches_PT = PatchExtraction.patch_extraction_thres(img_CT, img_PT, thres)

    # Normalize the inputs
    scaler_CT = StandardScaler()
    scaled_CT = scaler_CT.fit_transform(patches_CT)

    # Extract Features using network
    features = base_network.predict(scaled_CT)
    print(f"Shape of extracted features: {features.shape}")

    # Initialize clusters
    K_s = 20
    K_a = 5
    n_c_star = 100
    rand_samples = 15000
    loop_count = features.shape[0] // rand_samples
    print(f'{features.shape[0]} // {rand_samples} = {loop_count}')

    # randomly choose rand_samples
    for i in range(loop_count):
        loop_start = time.time()
        rand_ind = np.random.choice(features.shape[0], rand_samples)
        all_ind = np.arange(features.shape[0])

        print(f'Choose {rand_samples} samples randomly from {features.shape[0]} samples')
        cluster = ClusterInitialization.Clusters(features[rand_ind], rand_ind, n_c_star,
                                                 K_s=K_s, K_a=K_a)

        # Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
        dummy_gt_train = np.zeros((rand_samples, 151))

        # Merging Cluster Loop
        while cluster.is_finished():
            # Forward Pass
            cluster.aff_cluster_loop(eta=0.9)

            # Backward Pass
            H = model_triplet.fit(
                x=[patches_CT.numpy()[cluster.index], cluster.labels.astype('float32')],
                y=dummy_gt_train,
                batch_size=batch_size_per_replica,
                epochs=20,
                callbacks=[callback])

            # Update Features
            features_updated = base_network.predict([patches_CT, patches_PT])
            cluster.update_cluster(features_updated)

        # Exclude used samples
        features = features[np.setdiff1d(all_ind, rand_ind)]
        loop_end = time.time()

        num_exp += 1
        loop_msg = f'Loop #{num_exp} end. Elapsed Time: {loop_end - loop_start}\n'
        f.write(loop_msg)
        print(loop_msg)

now = datetime.now()
base_network.save_weights(dir_model)
time_end = time.time()

finish_msg = f'Training Finished! Elapsed Time: {time_end - time_start}'
f.write(finish_msg)
print(finish_msg)
f.close()
