#!/usr/bin/env python
# coding: utf-8
""" Training Autoencoder
    Make convolution autoencoder for comparison
"""
import tensorflow as tf
import os
from Network import NetworkKeras
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from Extraction import PatchExtraction


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

# FIXME : Parameters for training
path = './Training'
thres = 80
epochs = 30
batch_size = 128

# Extract Patches
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]

patient_list = os.listdir(path)
print(f'Number of patients: {len(patient_list)}')

# Initialize Autoencoder
input_shape = (17 * 17)
embedding_size = 150
AE_CT = NetworkKeras.create_autoencoder(input_shape)
AE_PT = NetworkKeras.create_autoencoder(input_shape)

input_CT = tf.keras.Input(shape=(17 * 17), name='Input_CT')
input_PT = tf.keras.Input(shape=(17 * 17), name='Input_PT')

AE_CT.summary()

callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
opt = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=5e-5)
AE_CT.compile(loss='binary_crossentropy', optimizer=opt)
AE_PT.compile(loss='binary_crossentropy', optimizer=opt)

# Make log file
now = datetime.now()
dir_model = f"./model/{now.strftime('%Y%m%d_%H%M%S')}/"
os.makedirs(dir_model)
os.makedirs(f'{dir_model}/CT')
os.makedirs(f'{dir_model}/PT')
f = open(f"{dir_model}log.txt", "w")

# Initialize parameters
num_exp = 0

for patient in patient_list:
    loop_start = time.time()

    # Make Patches
    img_CT, img_PT = PatchExtraction.stackImages(f'{path}/{patient}/', ind_CT, ind_PT)
    patches_CT, patches_PT = PatchExtraction.patch_extraction_thres(img_CT, img_PT, thres)

    # Normalize the inputs
    scaler_CT = StandardScaler()
    scaled_CT = scaler_CT.fit_transform(patches_CT)
    scaler_PT = StandardScaler()
    scaled_PT = scaler_PT.fit_transform(patches_PT)

    # Training
    AE_CT.fit(scaled_CT, scaled_CT,
              epochs=epochs, batch_size=batch_size,
              verbose=1, shuffle=True)
    AE_PT.fit(scaled_PT, scaled_PT,
              epochs=epochs, batch_size=batch_size,
              verbose=1, shuffle=True)

    loop_end = time.time()

    num_exp += 1
    loop_msg = f'Loop #{num_exp} end. Elapsed Time: {loop_end - loop_start}\n'
    f.write(loop_msg)
    print(loop_msg)

now = datetime.now()
AE_CT.save_weights(f'{dir_model}/CT/')
AE_PT.save_weights(f'{dir_model}/PT/')
time_end = time.time()

finish_msg = f'Training Finished! Elapsed Time: {time_end - time_start}'
f.write(finish_msg)
print(finish_msg)
f.close()
