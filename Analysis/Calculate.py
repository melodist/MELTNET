#!/usr/bin/env python
# coding: utf-8
""" Calculate Confusion Matrix
    Make confusion matrix for nodule detection
    1. Load feature index and offset value
    2. Overlay the feature image and mask image
    3. Make confusion matrix
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import time


def make_CM(path_result, path_ref):
    """ Make confusion matrix using mask generated from PT/CT image
        Confusion matrix of each slices will be recorded in one csv file
        This function makes 'CM_hhdd_HHMM.csv' in root directory
        CAUTION: index.csv file should be located in root directory.

    Input
    ______
    path_result : path of root directory for nodule detection image
    path_ref : path of root directory for reference image

    Output
    ______
    """

    # Read index from csv
    index = pd.read_csv(f'{path_result}/index.csv')
    patient_name = index['patient_name']
    feature_index = index['feature_index']
    offset = index['offset']

    # Find slice number of masks
    data = []
    for i, addr_maskfd in enumerate(patient_name):
        folder_mask = os.path.join(path_ref, addr_maskfd)
        list_mask = os.listdir(folder_mask)[:-1]
        folder_result = os.path.join(path_result, addr_maskfd)

        # Calculate confusion matrix
        for file_mask in list_mask:
            addr_mask = os.path.join(folder_mask, file_mask)
            mask = cv2.imread(addr_mask, cv2.IMREAD_GRAYSCALE)
            crop = np.array([[230, 380], [150, 370]])
            mask_crop = mask[crop[0, 0]:crop[0, 1], crop[1, 0]:crop[1, 1]]
            mask_crop[mask_crop > 0] = 255

            file_num = str(int(file_mask[5:8]) - offset[i])
            file_roi = f'Mask/Mask_I{file_num}0.png'
            file_roi_PT = f'Mask/MaskPT_I{file_num}0.png'
            file_feature = f'Features/Features_{feature_index[i]}_I{file_num}0.png'

            addr_roi = os.path.join(folder_result, file_roi)
            addr_roi_PT = os.path.join(folder_result, file_roi_PT)
            addr_feature = os.path.join(folder_result, file_feature)

            roi = cv2.imread(addr_roi, cv2.IMREAD_GRAYSCALE)
            roi_PT = cv2.imread(addr_roi_PT, cv2.IMREAD_GRAYSCALE)
            feature = cv2.imread(addr_feature, cv2.IMREAD_GRAYSCALE)
            result1 = cv2.bitwise_and(roi, feature)
            result = cv2.bitwise_and(result1, roi_PT)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(mask_crop.flatten() / 255, result.flatten() / 255).ravel()
            data.append([addr_maskfd, file_num, tp, fp, fn, tn])

    df = pd.DataFrame(data=data,
                      columns=['PatientName', 'SliceNumber', 'TP', 'FP', 'FN', 'TN'])

    # Save result to csv file
    hm = time.strftime('%H%M')
    result_date = os.path.split(path_result)[-1]
    df.to_csv(f'{path_result}/CM_{result_date}_{hm}.csv')


def make_CM_single(path_result, path_ref):
    """ Make confusion matrix using mask generated from CT image only
        Confusion matrix of each slices will be recorded in one csv file
        This function makes 'CM_hhdd_HHMM.csv' in root directory
        CAUTION: index.csv file should be located in root directory.

    Input
    ______
    path_result : path of root directory for nodule detection image
    path_ref : path of root directory for reference image

    Output
    ______
    """

    # Read index from csv
    index = pd.read_csv(f'{path_result}/index.csv')
    patient_name = index['patient_name']
    feature_index = index['feature_index']
    offset = index['offset']

    # Find slice number of masks
    data = []
    for i, addr_maskfd in enumerate(patient_name):
        folder_mask = os.path.join(path_ref, addr_maskfd)
        list_mask = os.listdir(folder_mask)[:-1]
        folder_result = os.path.join(path_result, addr_maskfd)

        # Calculate confusion matrix for single model
        for file_mask in list_mask:
            addr_mask = os.path.join(folder_mask, file_mask)
            mask = cv2.imread(addr_mask, cv2.IMREAD_GRAYSCALE)
            crop = np.array([[230, 380], [150, 370]])
            mask_crop = mask[crop[0, 0]:crop[0, 1], crop[1, 0]:crop[1, 1]]
            mask_crop[mask_crop > 0] = 255

            file_num = str(int(file_mask[5:8]) - offset[i])
            file_roi = f'Mask/Mask_I{file_num}0.png'
            file_feature = f'Features/Features_{feature_index[i]}_I{file_num}0.png'

            addr_roi = os.path.join(folder_result, file_roi)
            addr_feature = os.path.join(folder_result, file_feature)

            roi = cv2.imread(addr_roi, cv2.IMREAD_GRAYSCALE)
            feature = cv2.imread(addr_feature, cv2.IMREAD_GRAYSCALE)
            result = cv2.bitwise_and(roi, feature)
            #         plt.imshow(result, cmap='gray')

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(mask_crop.flatten() / 255, result.flatten() / 255).ravel()
            data.append([addr_maskfd, file_num, tp, fp, fn, tn])

    df = pd.DataFrame(data=data,
                      columns=['PatientName', 'SliceNumber', 'TP', 'FP', 'FN', 'TN'])

    # Save result to csv file
    hm = time.strftime('%H%M')
    result_date = os.path.split(path_result)[-1]
    df.to_csv(f'{path_result}/CM_{result_date}_{hm}.csv')


def make_CM_nomask(path_result, path_ref):
    """ Make confusion matrix without mask
        Confusion matrix of each slices will be recorded in one csv file
        This function makes 'CM_hhdd_HHMM.csv' in root directory
        CAUTION: index.csv file should be located in root directory.

    Input
    ______
    path_result : path of root directory for nodule detection image
    path_ref : path of root directory for reference image

    Output
    ______
    """

    # Read index from csv
    index = pd.read_csv(f'{path_result}/index.csv')
    patient_name = index['patient_name']
    feature_index = index['feature_index']
    offset = index['offset']

    # Find slice number of masks
    data = []
    for i, addr_maskfd in enumerate(patient_name):
        folder_mask = os.path.join(path_ref, addr_maskfd)
        list_mask = os.listdir(folder_mask)[:-1]
        folder_result = os.path.join(path_result, addr_maskfd)

        # Calculate confusion matrix without masks
        for file_mask in list_mask:
            addr_mask = os.path.join(folder_mask, file_mask)
            mask = cv2.imread(addr_mask, cv2.IMREAD_GRAYSCALE)
            crop = np.array([[230, 380], [150, 370]])
            mask_crop = mask[crop[0, 0]:crop[0, 1], crop[1, 0]:crop[1, 1]]
            mask_crop[mask_crop > 0] = 255

            file_num = str(int(file_mask[5:8]) - offset[i])
            file_feature = f'Features/Features_{feature_index[i]}_I{file_num}0.png'
            addr_feature = os.path.join(folder_result, file_feature)
            feature = cv2.imread(addr_feature, cv2.IMREAD_GRAYSCALE)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(mask_crop.flatten() / 255, feature.flatten() / 255).ravel()
            data.append([addr_maskfd, file_num, tp, fp, fn, tn])

    df = pd.DataFrame(data=data,
                      columns=['PatientName', 'SliceNumber', 'TP', 'FP', 'FN', 'TN'])

    # Save result to csv file
    hm = time.strftime('%H%M')
    result_date = os.path.split(path_result)[-1]
    df.to_csv(f'{path_result}/CM_{result_date}_{hm}.csv')


if __name__ == '__main__':
    make_CM(sys.argv[1], sys.argv[2])
