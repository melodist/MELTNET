"""Utils
    Functions for postprocessing of nodule detection

"""

import numpy as np
import cv2
from skimage.filters import threshold_multiotsu


def makeMask(img_CT):
    """Make Lung mask using CT image
        1. Remove corner and Binarize image using img_fill function
        2. Make contours using openCV
        3. Select 1 or 2 largest contour
        4. Make mask using contours

    Input
    ______
    img_CT : 2-D matrix of CT image

    Output
    ______
    mask : 2-D matrix of lung mask
    """

    # Initialize Kernels
    kernel3 = np.ones((3, 3), np.uint8)

    # Thresholding
    thresh = img_fill(img_CT, 210)
    result1 = cv2.morphologyEx(255 - thresh, cv2.MORPH_CLOSE, kernel3, iterations=3)

    # Finding Contour
    contours, hierarchy = cv2.findContours(result1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_contours.append(area)

    area_contours = np.array(area_contours)
    sort_contours = np.argsort(area_contours)

    # Exception
    if sort_contours.shape[0] < 2:
        contour_lung = contours
    elif area_contours[sort_contours[-1]] > area_contours[sort_contours[-2]] * 50:
        contour_lung = [contours[sort_contours[-1]]]
    else:
        contour_lung = [contours[sort_contours[-1]], contours[sort_contours[-2]]]

    mask = np.zeros(img_CT.shape, dtype='uint8')
    cv2.drawContours(mask, contour_lung, -1, (255), thickness=cv2.FILLED)

    return mask


def makeMask_PT(img_PT, i):
    """Extract nodule candidate using PT image
        1. Apply blur to image
        2. Find thresholds using multi-otsu thresholding
        3. Binarize image using ith threshold at step 2

    Input
    ______
    img_PT : 2-D matrix of PT image
    i : threshold level

    Output
    ______
    mask : 2-D matrix of nodule candidate image
    """

    im_blur = cv2.GaussianBlur(img_PT, (5, 5), 0)
    thresholds = threshold_multiotsu(im_blur, classes=5)
    ret, mask = cv2.threshold(im_blur, thresholds[-i], 255,
                              cv2.THRESH_BINARY)
    return mask


def img_fill(im_in, n):
    """ Binarize the image and fill the corner using floodfill algorithm
    
    Input
    ______
    im_in :  2-D input image
    n : binary image threshold

    Output
    ______
    im_floodfill : 2-D filled image
    """

    im_blur = cv2.GaussianBlur(im_in, (5, 5), 0)
    th, im_th = cv2.threshold(im_blur, n, 255, cv2.THRESH_BINARY);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    cv2.floodFill(im_floodfill, mask, (0, h - 1), 255);
    cv2.floodFill(im_floodfill, mask, (w - 1, 0), 255);
    cv2.floodFill(im_floodfill, mask, (w - 1, h - 1), 255);

    return im_floodfill
