#!/usr/bin/env python
# coding: utf-8
"""
Patch Extraction
Extract Patches from images.
Apply window to remove bed, etc.
"""

# Extract Patches from images
import os
import numpy as np
import tensorflow as tf
import cv2


def stackImages(path, ind_CT, ind_PT=-1):
    """Make PT/CT images into 3d matrix

    Input
    ______
    path : Path of root folder of images
    ind_CT : tuple for winodowing CT image.
        ex) ind_ct = ([width1, width2], [height1, height2])
    ind_PT : tuple for windowing PT image. If there is no value, function only returns img_CT.

    Output
    ______
    img_CT : 3-D matrix for CT images. [images, width, height]
    img_PT : 3-D matrix for PT images. [images, width, height]
    """
    if ind_PT == -1:
        # Make patches for CT only
        img_CT = np.zeros((1, ind_CT[0][1] - ind_CT[0][0],
                           ind_CT[1][1] - ind_CT[1][0]))
        for root, dirs, files in os.walk(path):
            files.sort()
            for file in files:
                filename = os.path.join(root, file)
                modality = filename.split(os.sep)[-2]

                if 'CT' in modality:
                    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                    # Apply windows
                    img_win = img[ind_CT[0][0]:ind_CT[0][1], ind_CT[1][0]:ind_CT[1][1]]
                    img_CT = np.vstack([img_CT, img_win[np.newaxis, :, :]])

        img_CT = np.delete(img_CT, 0, 0)
        img_CT = img_CT[:, :, :, np.newaxis]

        print(f'img_CT.shape: {img_CT.shape}')

        return img_CT

    else:
        img_CT = np.zeros((1, ind_CT[0][1] - ind_CT[0][0],
                           ind_CT[1][1] - ind_CT[1][0]))
        img_PT = np.zeros((1, ind_PT[0][1] - ind_PT[0][0],
                           ind_PT[1][1] - ind_PT[1][0]))
        for root, dirs, files in os.walk(path):
            files.sort()
            for file in files:
                filename = os.path.join(root, file)
                modality = filename.split(os.sep)[-2]

                if 'CT' in modality:
                    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                    # Apply windows
                    img_win = img[ind_CT[0][0]:ind_CT[0][1], ind_CT[1][0]:ind_CT[1][1]]
                    img_CT = np.vstack([img_CT, img_win[np.newaxis, :, :]])
                else:
                    img = 255 - cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                    # Apply windows
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
                    img_win = img[ind_PT[0][0]:ind_PT[0][1], ind_PT[1][0]:ind_PT[1][1]]
                    img_PT = np.vstack([img_PT, img_win[np.newaxis, :, :]])

        img_CT = np.delete(img_CT, 0, 0)
        img_PT = np.delete(img_PT, 0, 0)
        img_CT = img_CT[:, :, :, np.newaxis]
        img_PT = img_PT[:, :, :, np.newaxis]
        print(f'img_CT.shape: {img_CT.shape}')
        print(f'img_PT.shape: {img_PT.shape}')

        return img_CT, img_PT


def patch_extraction_thres(img_CT, img_PT=-1, thres=0):
    """Extract patches for training

    Input
    ______
    img_CT : 3-D matrix for CT images. [images, width, height]
    img_PT : 3-D matrix for PT images. [images, width, height]
            If there is no value, function only returns patches_CT.
    thres : parameter for thresholding by image intensity. Default value is 0.

    Output
    ______
    patches_CT : 4-D tensor for CT images. [patches, 17*17]
    patches_PT : 4-D tensor for PT images. [patches, 17*17]
    """
    if img_PT == -1:
        patches_all_CT = tf.extract_image_patches(img_CT, ksizes=[1, 17, 17, 1],
                                                  strides=[1, 5, 5, 1],
                                                  rates=[1, 1, 1, 1], padding='SAME')
        patches_all_CT = tf.reshape(patches_all_CT, [-1, 17 * 17])

        patches_center = patches_all_CT.numpy()[:, 144]
        patches_thres = patches_center > thres
        patches_CT = patches_all_CT[patches_thres]

        print(f"Patch Extraction Completed: {patches_CT.shape}")

        return patches_CT

    else:
        patches_all_CT = tf.extract_image_patches(img_CT, ksizes=[1, 17, 17, 1],
                                                  strides=[1, 5, 5, 1],
                                                  rates=[1, 1, 1, 1], padding='SAME')
        patches_all_CT = tf.reshape(patches_all_CT, [-1, 17 * 17])

        patches_all_PT = tf.extract_image_patches(img_PT, ksizes=[1, 17, 17, 1],
                                                  strides=[1, 5, 5, 1],
                                                  rates=[1, 1, 1, 1], padding='SAME')
        patches_all_PT = tf.reshape(patches_all_PT, [-1, 17 * 17])

        patches_center = patches_all_CT.numpy()[:, 144]
        patches_thres = patches_center > thres
        patches_CT = patches_all_CT[patches_thres]
        patches_PT = patches_all_PT[patches_thres]

        print(f"Patch Extraction Completed: {patches_CT.shape}")

        return patches_CT, patches_PT
