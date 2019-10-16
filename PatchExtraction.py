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

# In[91]:


# Extract Patches from images
import os
import glob
import numpy as np
import cv2

# Initialize Parameters

# Initialize Array
ind_CT = [[230, 380], [150, 370]]
ind_PT = [[66, 106], [43, 103]]

path = './PET-CT Images/00006347 KIM SE HO'
# Extrach path
def stackImages(path, ind_CT, ind_PT):
    img_CT = np.zeros((ind_CT[0][1]-ind_CT[0][0], 
                       ind_CT[1][1]-ind_CT[1][0]))
    img_PT = np.zeros((ind_PT[0][1]-ind_PT[0][0], 
                       ind_PT[1][1]-ind_PT[1][0]))
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            modality = filename.split(os.sep)[-2]

            if 'CT' in modality:
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # Apply windows
                img_win = img[ind_CT[0][0]:ind_CT[0][1],
                             ind_CT[1][0]:ind_CT[1][1]]
                img_CT = np.dstack((img_CT, img_win))
            else:
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                img_win = img[ind_PT[0][0]:ind_PT[0][1],
                              ind_PT[1][0]:ind_PT[1][1]]
                img_PT = np.dstack((img_PT, img_win))
                  
    img_CT = np.delete(img_CT, 0, 2)
    img_PT = np.delete(img_PT, 0, 2)
    print(f'img_CT.shape: {img_CT.shape}')
    print(f'img_PT.shape: {img_PT.shape}')
    
    return img_CT, img_PT


# In[94]:


# Read images from path
img = cv2.imread('./PET-CT Images/02215854 LEE SOON HWA/CT_lung/I1240.png')
cv2.imshow('img', img_CT[:,:,5].astype('uint8'))
while(True):
    k = cv2.waitKey(33)
    if k == -1:
        continue
    else:
        break
cv2.destroyWindow('img')

