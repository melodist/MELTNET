""" Test Autoencoder
    1. Extract Features using trained network
    2. Using K-means to classify the patches
    3. Merging Patches
"""
import tensorflow as tf
import numpy as np
from Network import NetworkKeras
import cv2
import os
import sys
import time
from datetime import datetime
from Evaluate import ImageProcessing
from Extraction import PatchExtraction
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def test_model(path_model, path_image, ind_CT, ind_PT, num_labels=5):
    """Test model using images in path_image
        This function makes result directory in the path:
         ./Results_yymmdd_HHMMSS'/
         Result directory has 3 subdirectory.
            CT : CT images used for test
            PT : PT images used for test
            Features : It has # of num_labels subdirectory.
                Each subdirectory has segmented result for each slices of patients.
    Input
    ______
    path_model : path of trained model
    path_image : path of images for test
    ind_CT : tuple for winodowing CT image.
        ex) ind_ct = ([width1, width2], [height1, height2])
    ind_PT : tuple for winodowing PT image.
    num_labels : number of clusters for K-means clustering. Default value is 5.

    Output
    ______
    """

    tf.enable_eager_execution()

    time_start = time.time()
    # Extract Features using trained network
    # Load model
    input_shape = (17 * 17)

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

    # Make Results Folder
    now = datetime.now()
    path_result = f"./Results_{now.strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(path_result)

    # Print Patients Number
    patient_dir = os.listdir(path_image)
    print(f'Patients Number: {len(patient_dir)}')

    for path_patient in patient_dir:
        addr_patient = f'{path_image}/{path_patient}/'
        path_files = path_result + path_patient + '/'
        os.makedirs(path_files)
        os.makedirs(f'{path_files}CT/')
        os.makedirs(f'{path_files}PT/')
        os.makedirs(f'{path_files}Features/')

        img_CT, img_PT = PatchExtraction.stackImages(addr_patient, ind_CT, ind_PT)
        patches_CT, patches_PT = PatchExtraction.patch_extraction_thres(img_CT, img_PT, 0)

        # Normalize the inputs
        scaler_CT = StandardScaler()
        scaled_CT = scaler_CT.fit_transform(patches_CT)
        scaler_PT = StandardScaler()
        scaled_PT = scaler_PT.fit_transform(patches_PT)

        # Extract Features
        print(f"Extract Features...")
        features_CT = feature_extractor_CT.predict(scaled_CT, steps=1)
        features_PT = feature_extractor_PT.predict(scaled_PT, steps=1)
        features = np.hstack((features_CT, features_PT))

        # Using K-means
        print(f"K-means Clustering...")
        model_k_means = KMeans(n_clusters=num_labels, random_state=0)
        model_k_means.fit(features)

        # Merging Patches
        num_x = 44
        num_y = 30
        stride = 5

        label_predict = model_k_means.fit_predict(features)
        label_predict_batch = label_predict.reshape((-1, num_y * num_x))

        # Extract File Names
        for root, dirs, files in os.walk(os.path.join(path_image, path_patient)):
            file_list = files
            file_list.sort()

        print(f'Merging Patches...')
        for i, filename in enumerate(file_list):
            mask = ImageProcessing.project_patches(label_predict_batch[i, :], num_labels, num_y, num_x, stride, 5)
            for j in range(num_labels):
                ImageProcessing.save_image(mask[:, :, j], f'./Features/Features_{j}_' + filename, path_files)
            # save original image as reference
            cv2.imwrite(path_files + 'CT/' + filename, img_CT[i, :, :, 0])
            cv2.imwrite(path_files + 'PT/' + filename, img_PT[i, :, :, 0])

    time_end = time.time()
    print(f"Test Finished! Elapsed time: {time_end - time_start:.2f}")


if __name__ == '__main__':
    test_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
