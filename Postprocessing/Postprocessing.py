#!/usr/bin/env python
# coding: utf-8
"""Postprocessing
    Remove the contour that indicates inner organs.

"""

import numpy as np
import os
import sys
import glob
import cv2
import pandas as pd
import shutil
from . import utils


def load_data(path_result):
    """ Load the list of image folder and following feature index.
        CAUTION: index.csv file should be located in root directory.

    Input
    ______
    path_result : path of root directory

    Output
    ______
    list_folder : list of patients folders
    feature_index : following feature index for each patient
    """
    list_all = glob.glob(f'{path_result}/*')
    list_folder = list(filter(os.path.isdir, list_all))
    print(list_folder)

    # Read index from csv
    index = pd.read_csv(f'{path_result}/index.csv')
    feature_index = index['feature_index']

    return list_folder, feature_index


def postprocessing_no_mask(path_result):
    """ Nodule detection without using any mask
        This function makes subdirectory in each patient directory
            Overlay_No : Original CT images + Detected nodule images

    Input
    ______
    path_result : path of root directory

    Output
    ______
    """

    list_folder, feature_index = load_data(path_result)

    for i, addr_results in enumerate(list_folder):

        list_index = os.listdir(addr_results + '/CT')

        list_file = os.listdir(addr_results)
        list_png = [f for f in list_file if f.endswith(".png")]
        for f in list_png:
            os.remove(addr_results + '/' + f)

        # Initialize Overlay folder
        if os.path.isdir(addr_results + '/Overlay_No'):
            shutil.rmtree(addr_results + '/Overlay_No')
        os.mkdir(addr_results + '/Overlay_No')

        for file_index in list_index:
            img_CT = cv2.imread(addr_results + '/CT/' + file_index)
            filename = f'Features_{feature_index[i]}_{file_index}'
            addr_file = os.path.join(addr_results, 'Features', filename)
            image = cv2.imread(addr_file, cv2.IMREAD_GRAYSCALE)

            # Make overlay violet
            img_nodule = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            img_nodule[:, :, 1:2] = 0

            # Blend Images
            dst = cv2.addWeighted(img_CT, 0.6, img_nodule, 0.4, 0)

            # Save Images
            cv2.imwrite(f'{addr_results}/Overlay_No/Overlay_No_{file_index}', dst)


def postprocessing(root_result):
    """ Nodule detection using mask generated from CT/PT image
        This function makes subdirectories in each patient directory
            Overlay : Original CT images + Detected nodule images
            Mask : Lung Region Mask generated from CT image
            MaskPT : Nodule Candidate Mask generated from PT image

    Input
    ______
    path_result : path of root directory

    Output
    ______
    """

    list_folder, feature_index = load_data(root_result)

    for i, addr_results in enumerate(list_folder):

        list_index = os.listdir(addr_results + '/CT')

        list_file = os.listdir(addr_results)
        list_png = [f for f in list_file if f.endswith(".png")]
        for f in list_png:
            os.remove(addr_results + '/' + f)

        # Initialize Overlay folder
        if os.path.isdir(addr_results + '/Overlay'):
            shutil.rmtree(addr_results + '/Overlay')
        os.mkdir(addr_results + '/Overlay')

        # Initialize Mask folder
        if os.path.isdir(addr_results + '/Mask'):
            shutil.rmtree(addr_results + '/Mask')
        os.mkdir(addr_results + '/Mask')

        for file_index in list_index:
            filename = f'Features_{feature_index[i]}_{file_index}'
            addr_file = os.path.join(addr_results, 'Features', filename)
            image = cv2.imread(addr_file, cv2.IMREAD_GRAYSCALE)

            # Make CT masks
            img_CT = cv2.imread(addr_results + '/CT/' + file_index)
            img_CT_gray = cv2.imread(addr_results + '/CT/' + file_index, cv2.IMREAD_GRAYSCALE)
            mask_CT = utils.makeMask(img_CT_gray)
            cv2.imwrite(f'{addr_results}/Mask/Mask_{file_index}', mask_CT)

            # Make PT masks
            img_PT = cv2.imread(addr_results + '/PT/' + file_index, cv2.IMREAD_GRAYSCALE)
            mask_PT = utils.makeMask_PT(img_PT, 2)
            cv2.imwrite(f'{addr_results}/Mask/MaskPT_{file_index}', mask_PT)

            # Bitwise calculation
            nodule = cv2.bitwise_and(image, mask_CT)
            nodule2 = cv2.bitwise_and(nodule, mask_PT)

            # Make nodule violet
            img_nodule = np.repeat(nodule2[:, :, np.newaxis], 3, axis=2)
            img_nodule[:, :, 1:2] = 0

            # Blend Images
            dst = cv2.addWeighted(img_CT, 0.6, img_nodule, 0.4, 0)

            # Save Overlay Images
            cv2.imwrite(f'{addr_results}/Overlay/Overlay_{file_index}', dst)


def postprocessing_single(root_result):
    """ Nodule detection using mask generated from CT image
        This function makes subdirectories in each patient directory
            Overlay : Original CT images + Detected nodule images
            Mask : Lung Region Mask generated from CT image

    Input
    ______
    path_result : path of root directory

    Output
    ______
    """

    list_folder, feature_index = load_data(root_result)

    # Remove organs for single model
    for i, addr_results in enumerate(list_folder):

        list_index = os.listdir(addr_results + '/CT')

        list_file = os.listdir(addr_results)
        list_png = [f for f in list_file if f.endswith(".png")]
        for f in list_png:
            os.remove(addr_results + '/' + f)

        # Initialize Overlay folder
        if os.path.isdir(addr_results + '/Overlay'):
            shutil.rmtree(addr_results + '/Overlay')
        os.mkdir(addr_results + '/Overlay')

        # Initialize Mask folder
        if os.path.isdir(addr_results + '/Mask'):
            shutil.rmtree(addr_results + '/Mask')
        os.mkdir(addr_results + '/Mask')

        for file_index in list_index:
            filename = f'Features_{feature_index[i]}_{file_index}'
            addr_file = os.path.join(addr_results, 'Features', filename)
            image = cv2.imread(addr_file, cv2.IMREAD_GRAYSCALE)

            # Make CT masks
            img_CT = cv2.imread(addr_results + '/CT/' + file_index)
            img_CT_gray = cv2.imread(addr_results + '/CT/' + file_index, cv2.IMREAD_GRAYSCALE)
            mask_CT = utils.makeMask(img_CT_gray)
            cv2.imwrite(f'{addr_results}/Mask/Mask_{file_index}', mask_CT)

            # Bitwise calculation
            nodule = cv2.bitwise_and(image, mask_CT)

            # Make overlay violet
            img_nodule = np.repeat(nodule[:, :, np.newaxis], 3, axis=2)
            img_nodule[:, :, 1:2] = 0

            # Blend Images
            dst = cv2.addWeighted(img_CT, 0.6, img_nodule, 0.4, 0)

            # Save Overlay Images
            cv2.imwrite(f'{addr_results}/Overlay/Overlay_{file_index}', dst)


if __name__ == '__main__':
    postprocessing(sys.argv[1])
