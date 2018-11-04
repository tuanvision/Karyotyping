import numpy as np
import cv2
from skimage.morphology import reconstruction, remove_small_objects, remove_small_holes
from os.path import dirname, abspath
from matplotlib import pyplot as plt
from util import isclose


# ========== MAIN METHODS ==========

def threshold(gray, mode=None):
    if mode == 'edge':
        return threshold_by_edge(gray)
    if mode == 'grisan':
        return grisan_local_threshold(gray)
    if mode == 'adaptive otsu':
        return adaptive_otsu_threshold(gray)


'''
    Reference: Digital Image Processing 2nd, page 750
'''
def threshold_by_edge(gray):
    # Step 1: Compute edge image
    edge = edge_detection(gray, 'laplacian')
    show_image(edge)

    # Step 2: Threshold edge image
    ret, thresh = cv2.threshold(edge, thresh=histogram_percentile_value(edge, 0.05), maxval=255, type=cv2.THRESH_BINARY_INV)
    # ret, thresh = cv2.threshold(edge, thresh=np.amax(edge) / 100, maxval=255, type=cv2.THRESH_BINARY_INV)
    show_image(thresh)


    # Step 3: Compute histogram of orignal image masked by thresholded edge image
    masked = gray.copy()
    masked[thresh != 255] = 0
    show_image(masked)
    ret2, thresh2 = cv2.threshold(masked, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 4: Use threshold value from step 3 to threshold the original image
    ret3, thresh3 = cv2.threshold(gray, thresh=ret2, maxval=255, type=cv2.THRESH_BINARY_INV)
    show_image(thresh3)

    return thresh3

# reference:  Automatic segmentation and disentangling of chromosomes in Q-band prometaphase images
def grisan_local_threshold(gray, grid_size=350):
    height, width = gray.shape
    matrix_height = int(height/grid_size)
    matrix_width = int(width/grid_size)
    matrix_threshold = [[get_grid_threshold_value(gray, i, j, grid_size) for j in range(matrix_width)] for i in range(matrix_height)]
    matrix_threshold = np.asarray(matrix_threshold)
    gray_threshold = cv2.resize(matrix_threshold, (width, height), interpolation=cv2.INTER_LINEAR)
    threshold = np.zeros_like(gray)
    threshold[gray > gray_threshold] = 255
    threshold[gray <= gray_threshold] = 0
    return threshold

# reference: https://dsp.stackexchange.com/questions/2411/what-are-the-most-common-algorithms-for-adaptive-thresholding
def adaptive_otsu_threshold(gray, grid_size=10):
    threshold = np.zeros_like(gray)
    height, width = gray.shape    
    h_upper_bound = int(height/grid_size)
    w_upper_bound = int(width/grid_size)
    for i in range(h_upper_bound):
        for j in range(w_upper_bound):
            threshold[max(0, i - 1) * grid_size : i * grid_size][max(0, j - 1) * grid_size : j * grid_size] = get_grid_threshold(gray, i, j, grid_size)
    return threshold


def partial_otsu_threshold(gray, minval=0, maxval=255, dark_background=True):
    if not dark_background:
        gray = 255 - gray
    histogram = cv2.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=[0,256])
    histogram = [element[0] for element in histogram]
    ret = compute_otsu_value(histogram[minval : (maxval + 1)])
    ret, thresh = cv2.threshold(gray, ret, 255, cv2.THRESH_BINARY)
    return thresh

def variable_threshold(gray, kernel_size=(3, 3), alpha=20, beta=0.5):
    print(gray)
    height, width = gray.shape
    num_pixel_in_kernel = kernel_size[0] * kernel_size[1]

    # average kernel
    kernel = np.ones(kernel_size, np.float32) / num_pixel_in_kernel

    mean = cv2.filter2D(gray, ddepth=-1, kernel=kernel)
    print(mean)
    std = np.abs(gray - mean) / ((height * width) ** (1/2))
    print(std)
    ret = alpha * std + beta * mean
    print(ret)
    thresh = (gray > ret) * 1
    print(thresh)

    return thresh



# ========== UTILITY METHODS ==========

def get_grid_threshold_value(gray, i, j, grid_size):
    sub_gray = gray[max(0, i - 1) * grid_size : i * grid_size][max(0, j - 1) * grid_size : j * grid_size]
    ret, thresh = cv2.threshold(sub_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret

def get_grid_threshold(gray, i, j, grid_size):
    sub_gray = gray[max(0, i - 1) * grid_size : i * grid_size][max(0, j - 1) * grid_size : j * grid_size]
    ret, thresh = cv2.threshold(sub_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def edge_detection(gray, type='laplacian'):
    if type == 'laplacian':
        laplacian = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=1)
        unit8_laplacian = cv2.convertScaleAbs(laplacian)
        return unit8_laplacian
    if type == 'canny':
        return cv2.Canny(gray, threshold1=100, threshold2=200)
    raise ValueError('type is not supported!')

def histogram_percentile_value(gray, percentile=0.1):
    histogram = cv2.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=[0,256])
    num_value = np.sum(histogram)
    current_sum = 0
    for i in range(256):
        current_sum += histogram[i]
        if (current_sum < num_value * percentile):
            return current_sum

    return 1


# reference: 10.3.3 Digital Image Processing
def compute_otsu_value(histogram):
    total_pixels = sum(histogram)
    num_value = len(histogram)

    # calculate p_i
    probability = histogram / total_pixels

    # calculate P1_k
    curr = 0
    cumulative_probability = list()
    for i in range(num_value):
        curr += probability[i]
        cumulative_probability.append(curr)

    
    # calculate m_G
    individual_mean = [intensity * count for intensity, count in enumerate(histogram)]
    global_mean = sum(individual_mean)

    # calculate m_k
    curr_mean = 0
    cumulative_mean = list()
    for i in range(num_value):
        curr_mean += individual_mean[i]
        cumulative_mean.append(curr_mean)

    # calculate between-class variance sigma_B
    sigma_B = list()
    for i in range(num_value):
        denominator = (cumulative_probability[i] * (1 - cumulative_probability[i]))
        if isclose(denominator, 0):
            sigma = 0
        else:
            sigma = pow(global_mean * cumulative_probability[i] - cumulative_mean[i], 2) / denominator
        sigma_B.append(sigma)

    # calculate max k
    max_sigma_value = np.amax(np.asarray(sigma_B))
    sigma_maxs = list()
    for i in range(num_value):
        if isclose(sigma_B[i], max_sigma_value):
            sigma_maxs.append(i)

    return int(sum(sigma_maxs) / len(sigma_maxs))