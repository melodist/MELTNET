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
import pickle
from Extraction import PatchExtraction
from Cluster import ClusterInitialization, ClusterMerging
import NetworkTensorflow
import NetworkKeras
import TripletLossAdaptedFromTF

tf.enable_eager_execution()

# Training Networks

# Extract Patches
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]
path = './Example'

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    img_CT, img_PT = PatchExtraction.stackImages(path, ind_CT, ind_PT)
    patches_CT = PatchExtraction.patch_extraction(img_CT)
    patches_PT = PatchExtraction.patch_extraction(img_PT)

    # Initialize Network
    input_shape = (17 * 17)
    embedding_size = 150
    base_network = NetworkKeras.create_base_network(input_shape, embedding_size)
    base_network.summary()

    # Extract Features using network
    features = base_network.predict([patches_CT, patches_PT])
    print(f"Shape of extracted features: {features.shape}")

    # Initialize clusters
    K_s = 20
    K_a = 5
    n_c_star = 100
    rand_samples = 10000

    rand_ind = np.random.choice(features.shape[0], rand_samples)  # randomly choose rand_samples
    cluster = ClusterInitialization.Clusters(features[rand_ind], n_c_star, K_s=K_s, K_a=K_a)

    # Save cluster to binary file
    with open('test_191027.pickle', 'wb') as f:
        pickle.dump(cluster, f)

    # # Initialize Network
    # batch_size = 128
    # input_CT = tf.keras.Input(shape=(17 * 17), name='Input_CT')
    # input_PT = tf.keras.Input(shape=(17 * 17), name='Input_PT')
    # input_labels = tf.keras.Input(shape=(1,), name='Input_label')
    #
    # embeddings = base_network([input_CT, input_PT])
    # labels_plus_embeddings = tf.keras.layers.concatenate([input_labels, embeddings])
    # model_triplet = tf.keras.Model(inputs=[input_CT, input_PT, input_labels],
    #                                outputs=labels_plus_embeddings)
    #
    # model_triplet.summary()
    # tf.keras.utils.plot_model(model_triplet, to_file='model_triplet.png',
    #                           show_shapes=True, show_layer_names=True)
    # opt = tf.keras.optimizers.Adam(lr=0.0001)
    # model_triplet.compile(loss=TripletLossAdaptedFromTF.triplet_loss_adapted_from_tf,
    #                       optimizer=opt)
    #
    # filepath = '{epoch:02d}-{val_loss:.4f}.hdf5'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath, monitor='val_loss', verbose=1, save_best_only=False, period=25)
    # callbacks_list = [checkpoint]
    #
    # # Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
    # dummy_gt_train = np.zeros((patches_CT.shape[0], 151))
    #
    # # Merging Cluster Loop
    # while cluster.is_finished():
    #     # Forward Pass
    #     cluster.aff_cluster_loop(eta=0.9)
    #
    #     # Backward Pass
    #     H = model_triplet.fit(
    #         x=[patches_CT, patches_PT, cluster.labels],
    #         y=dummy_gt_train,
    #         batch_size=batch_size,
    #         epochs=20,
    #         callbacks=callbacks_list)
