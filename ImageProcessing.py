import cv2
import os
import numpy as np
from sklearn.metrics import euclidean_distances


# Merging Patches
def merging_patches(labels, num_labels, num_y, num_x, stride):
    mask_image = np.zeros((num_y * stride, num_x * stride, num_labels))
    mesh = np.arange(num_y * num_x).reshape((num_y, num_x))

    for x in range(num_x):
        for y in range(num_y):
            mask_image[stride * y:stride * y + 17, stride * x:stride * x + 17, labels[mesh[y, x]]] += 1

    for i in range(num_labels):
        mask_image[:, :, i] = mask_image[:, :, i] / mask_image[:, :, i].max() * 255

    return mask_image


# Project Patches
def project_patches(labels, num_labels, num_y, num_x, stride):
    mask_image = np.zeros((num_y * stride, num_x * stride, num_labels))
    mesh = np.arange(num_y * num_x).reshape((num_y, num_x))

    # Calculate centroids of patches
    x_center = np.array(range(2, 220, 5))
    y_center = np.array(range(2, 150, 5))
    X, Y = np.meshgrid(x_center, y_center)
    centers = np.array([list(zip(x, y)) for x, y in zip(X, Y)])

    for x in x_center:
        for y in y_center:
            dists = euclidean_distances([[x, y]], centers.reshape(30 * 44, 2))

            # Reshape dists and find minimum index
            dists_reshape = dists.reshape(30, 44)
            x_min = np.where(dists.min() == dists_reshape)[1][0]
            y_min = np.where(dists.min() == dists_reshape)[0][0]

            mask_image[y - 2:y + 3, x - 2:x + 3, labels[mesh[y_min, x_min]]] += 1

    for i in range(num_labels):
        mask_image[:, :, i] = mask_image[:, :, i] / mask_image[:, :, i].max() * 255

    return mask_image


# Visualize Results
def save_image(image, filename, path):
    fileaddr = path + filename
    cv2.imwrite(fileaddr, image)


def ImageBlending(diraddr, feature_num):
    # Read Images
    ct_list = os.listdir(f'{diraddr}CT/')
    ct_list.sort()

    for i in ct_list:
        img_CT = cv2.imread(f'{diraddr}CT/{i}')
        img_PT = cv2.imread(f'{diraddr}PT/{i}')

        # Make mask
        img_mask_plane = make_mask(img_CT, img_PT)
        cv2.imwrite(f'{diraddr}Mask_{i}', img_mask_plane)

        for j in range(feature_num):
            img_result = cv2.imread(f'{diraddr}Features/Features_{j}_{i}')

            # Bitwise calculation
            img_result_plane = img_result[:, :, 0]
            img_overlay_plane = cv2.bitwise_and(img_result_plane, img_mask_plane)

            # Make overlay violet
            img_overlay = np.repeat(img_overlay_plane[:, :, np.newaxis], 3, axis=2)
            img_overlay[:, :, 1:2] = 0

            # Blend Images
            dst = cv2.addWeighted(img_CT, 0.6, img_overlay, 0.4, 0)

            # Save Images
            cv2.imwrite(f'{diraddr}/Overlay/Overlay_{j}_{i}', dst)

    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('Image Blending Finished')


def make_mask(img_CT, img_PT):
    gray = 255 - cv2.cvtColor(img_CT, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Finding Lung Contour using CT Images
    images, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_contours.append(area)

    area_contours = np.array(area_contours)
    sort_contours = np.argsort(area_contours)

    contour_lung = [contours[sort_contours[-1]], contours[sort_contours[-2]]]
    mask = np.zeros(img_CT.shape, np.uint8)
    cv2.drawContours(mask, contour_lung, -1, 255, thickness=cv2.FILLED)

    # Finding DMR using PT Images
    gray_PT = cv2.cvtColor(img_PT, cv2.COLOR_BGR2GRAY)
    ret_PT, thresh_PT = cv2.threshold(gray_PT, 200, 255, cv2.THRESH_BINARY)
    opener = np.ones((5, 5))
    thresh_PT_open = cv2.morphologyEx(thresh_PT, cv2.MORPH_OPEN, opener)

    # Make contours
    _, contours_PT, _ = cv2.findContours(thresh_PT_open, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate average value in contours
    kernel = np.ones((5, 5))

    thres_in = 230
    thres_lung = 190
    img_CT_plane = img_CT[:, :, 0]

    # Compensate the lung region
    print(f'contours number: {len(contours_PT)}')
    for i in range(len(contours_PT)):

        # Draw mask for a contour
        mask_cnt = np.zeros(img_CT.shape, dtype=np.uint8)
        cv2.drawContours(mask_cnt, contours_PT, i, 255, thickness=cv2.FILLED)
        mask_orig = mask_cnt[:, :, 0]

        # Expand contours using dilation
        mask_expanded = cv2.dilate(mask_orig, kernel, iterations=1)

        # Make DMR
        mask_orig_not = cv2.bitwise_not(mask_orig)
        mask_DMR = cv2.bitwise_and(mask_expanded, mask_orig_not)

        # Calculate average value of DMR
        DMR_point = np.where(mask_DMR > 0)

        merged = cv2.bitwise_and(img_CT_plane, mask_DMR)
        mean_DMR = np.mean(merged[DMR_point])
        # print(mean_DMR)

        dist_in = np.abs(thres_in - mean_DMR)
        dist_lung = np.abs(thres_lung - mean_DMR)

        result_total = mask[:, :, 0]
        # Find area has similarity with interstitial space of lung or parenchyma of lung
        if dist_in > dist_lung:
            M = cv2.moments(contours_PT[i])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            result = regiongrow(img_CT_plane, 255 * 0.1, (cy, cx))
            result_total = cv2.bitwise_or(result_total, result)

    result_expanded = cv2.morphologyEx(result_total, cv2.MORPH_CLOSE, kernel)

    return result_expanded


def region_growing(img, seed):
    # Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_threshold = 0.3
    region_size = 1
    intensity_difference = 0
    neighbor_points_list = []
    neighbor_intensity_list = []

    # Mean of the segmented region
    region_mean = img[seed]

    # Input image parameters
    height, width = img.shape
    image_size = height * width

    # Initialize segmented output image
    segmented_img = np.zeros((height, width, 1), np.uint8)

    # Region growing until intensity difference becomes greater than certain threshold
    while (intensity_difference < region_threshold) & (region_size < image_size):
        # Loop through neighbor pixels
        for i in range(4):
            # Compute the neighbor pixel position
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            # Boundary Condition - check if the coordinates are inside the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)

            # Add neighbor if inside and not already in segmented_img
            if check_inside:
                if segmented_img[x_new, y_new] == 0:
                    neighbor_points_list.append([x_new, y_new])
                    neighbor_intensity_list.append(img[x_new, y_new])
                    segmented_img[x_new, y_new] = 255

        # Add pixel with intensity nearest to the mean to the region
        distance = abs(neighbor_intensity_list - region_mean)
        pixel_distance = min(distance)
        index = np.where(distance == pixel_distance)[0][0]
        segmented_img[seed[0], seed[1]] = 255
        region_size += 1

        # New region mean
        region_mean = (region_mean * region_size + neighbor_intensity_list[index]) / (region_size + 1)

        # Update the seed value
        seed = neighbor_points_list[index]
        # Remove the value from the neighborhood lists
        neighbor_intensity_list[index] = neighbor_intensity_list[-1]
        neighbor_points_list[index] = neighbor_points_list[-1]

    return segmented_img


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enque(self, item):
        self.items.insert(0, item)

    def deque(self):
        return self.items.pop()

    def qsize(self):
        return len(self.items)

    def isInside(self, item):
        return item in self.items


def regiongrow(image, epsilon, start_point):
    Q = Queue()
    s = []

    x = start_point[0]
    y = start_point[1]

    Q.enque((x, y))

    image = image.astype(np.int16)
    mean_region = image[x, y]

    while not Q.isEmpty():

        t = Q.deque()
        x = t[0]
        y = t[1]

        if x < image.shape[0] - 1 and \
                abs(mean_region - image[x + 1, y]) <= epsilon:

            if not Q.isInside((x + 1, y)) and not (x + 1, y) in s:
                Q.enque((x + 1, y))

        if x > 0 and \
                abs(mean_region - image[x - 1, y]) <= epsilon:

            if not Q.isInside((x - 1, y)) and not (x - 1, y) in s:
                Q.enque((x - 1, y))

        if y < (image.shape[1] - 1) and \
                abs(mean_region - image[x, y + 1]) <= epsilon:
            if not Q.isInside((x, y + 1)) and not (x, y + 1) in s:
                Q.enque((x, y + 1))

        if y > 0 and \
                abs(mean_region - image[x, y - 1]) <= epsilon:

            if not Q.isInside((x, y - 1)) and not (x, y - 1) in s:
                Q.enque((x, y - 1))

        if t not in s:
            s.append(t)

        points = np.asarray(s)
        mean_region = image[points[:, 0], points[:, 1]].mean()

    mask = np.zeros(image.shape, dtype=np.uint8)

    for i in s:
        mask[i[0], i[1]] = 255

    return mask
