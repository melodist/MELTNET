""" Test JULE for single modality
    1. Extract Features using trained network
    2. Using K-means to classify the patches
    3. Merging Patches
"""

import tensorflow as tf
from Network import NetworkKeras
import cv2
import os
import sys
import time
import ImageProcessing
from Extraction import PatchExtraction
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def test_model(path_model, path_image, ind_CT, num_labels=5):
    """Test model using images in path_image
        This function makes result directory in the path:
         ./Results_yymmdd_HHMMSS'/
         Result directory has 2 subdirectory.
            CT : CT images used for test
            Features : It has # of num_labels subdirectory.
                Each subdirectory has segmented result for each slices of patients.
    Input
    ______
    path_model : path of trained model
    path_image : path of images for test
    ind_CT : tuple for winodowing CT image.
        ex) ind_ct = ([width1, width2], [height1, height2])
    num_labels : number of clusters for K-means clustering. Default value is 5.

    Output
    ______
    """

    tf.enable_eager_execution()
    time_start = time.time()

    # 1. Extract Features using trained network
    # Load model
    input_shape = (17 * 17)
    embedding_size = 150
    trained_model = NetworkKeras.create_single_network(input_shape, embedding_size)
    trained_model.load_weights(path_model)

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
        os.makedirs(f'{path_files}Features/')

        img_CT = PatchExtraction.stackImages(addr_patient, ind_CT)
        patches_CT = PatchExtraction.patch_extraction_thres(img_CT)

        # Normalize the inputs
        scaler_CT = StandardScaler()
        scaled_CT = scaler_CT.fit_transform(patches_CT)

        # Extract Features
        print(f"Extract Features...")
        features = trained_model.predict(scaled_CT, steps=1)

        # Using K-means
        print(f"K-Means Clustering...")
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

    time_end = time.time()
    print(f"Test Finished! Elapsed time: {time_end - time_start:.2f}")


if __name__ == '__main__':
    test_model(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
