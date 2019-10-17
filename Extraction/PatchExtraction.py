#!/usr/bin/env python
# coding: utf-8

# ## Patch Extraction
# Extract Patches from images  
# Apply window to remove bed, etc.  
# window_CT : 230:380, 150:370 (144, 216)  
# window_PT : 66:106, 43:103 (40, 60)  
#   
# Patch_CT : 17x17 stride 7  
# Patch_PT : 5x5 stride 2  
# 113 subjects  


# Extract Patches from images
import os
import numpy as np
import cv2


# Extract path
def stackImages(path, ind_ct, ind_pt):
    img_ct = np.zeros((1, ind_ct[0][1] - ind_ct[0][0],
                       ind_ct[1][1] - ind_ct[1][0]))
    img_pt = np.zeros((1, ind_ct[0][1] - ind_ct[0][0],
                       ind_ct[1][1] - ind_ct[1][0]))
    for root, dirs, files in os.walk(path):
        dir_list = [d for d in dirs]
        for file in files:
            filename = os.path.join(root, file)
            modality = filename.split(os.sep)[-2]

            if 'CT' in modality:
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # Apply windows
                img_win = img[ind_ct[0][0]:ind_ct[0][1], ind_ct[1][0]:ind_ct[1][1]]
                img_ct = np.vstack([img_ct, img_win[np.newaxis, :, :]])
            else:
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # Apply windows
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
                img_win = img[ind_ct[0][0]:ind_ct[0][1], ind_ct[1][0]:ind_ct[1][1]]
                img_pt = np.vstack([img_pt, img_win[np.newaxis, :, :]])

    img_ct = np.delete(img_ct, 0, 0)
    img_pt = np.delete(img_pt, 0, 0)
    img_ct = img_ct[:, :, :, np.newaxis]
    img_pt = img_pt[:, :, :, np.newaxis]
    print(f'img_ct.shape: {img_ct.shape}')
    print(f'img_pt.shape: {img_pt.shape}')

    return img_ct, img_pt
