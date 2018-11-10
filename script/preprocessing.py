import numpy as np
import cv2
from skimage.morphology import reconstruction, remove_small_objects, remove_small_holes
from os.path import dirname, abspath
from matplotlib import pyplot as plt
from util import show_image

image_dir = dirname(dirname(abspath("X"))) + "/data/"

def background_removal(image, stride=20, window_size=(100, 100), num_contour_threshold=3, acceptance_ratio=2, chromosome_type=None):
    # CONVERT IMAGE TO GRAYSCALE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show_image(gray, name='gray')

    # USE MEDIAN FILTER TO REMOVE NOISE
    median = cv2.medianBlur(gray, 5)
    # show_image(median)

    # # NORMALIZE HISTOGRAM
    # gray = histogram_equalization(gray, type="clahe")
    # show_image(gray, name="he")

    # # PERFORM UNSHARP MASKING
    # gray = unsharp_masking(gray)
    # show_image(gray, name="unsharp")

    # THRESHOLD BY OTSU'S METHOD
    ret, thresh = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
    # show_image(thresh, name="thresh")

    # GET CONTOURS
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image_with_contours = cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)

    # MASK EACH CONTOUR BY A NUMBER (ID)
    mask_contour = create_mask_contour(contours, shape=gray.shape)

    # CREATE COUNTING DICTIONARY OF HOW MANY VALID AND INVALID WINDOWS FOR EACH CONTOUR
    num_contour = len(contours)
    valid = {contour_id: 0 for contour_id in range(1, num_contour + 1)}
    in_valid = {contour_id: 0 for contour_id in range(1, num_contour + 1)}

    # SLIDE WINDOW
    for y in range(0, gray.shape[0], stride):
        for x in range(0, gray.shape[1], stride):
            # GET WINDOW
            arr_window = mask_contour[x: x + window_size[0], y: y + window_size[1]]

            # GET CONTOURS IN THE WINDOW 
            unique, counts = np.unique(arr_window, return_counts=True)         

            # UPDATE 2 DICTIONARIES 
            # A WINDOW IS VALID IF THE NUMBER OF CONTOURS IN THAT WINDOW IS GREATER THAN A THRESHOLD, ELSE INVALID.
            num_con_window = np.count_nonzero(unique)
            if num_con_window > num_contour_threshold :
                for con_id in unique:
                    if con_id == 0: 
                        continue
                    valid[con_id] = valid[con_id] + 1
            else:
                for con_id in unique:
                    if con_id == 0: 
                        continue
                    in_valid[con_id] = in_valid[con_id] + 1

    # CREATE A LIST OF CONTOURS TO REMOVE
    # A CONTOUR WILL BE REMOVED IF #VALID/#INVALID IS LOWER THAN A RATIO
    chosen_contours = list()
    for i in range(1, num_contour + 1):
        if valid[i] * acceptance_ratio >= in_valid[i]:
            chosen_contours.append(contours[i - 1])

    # REMOVE CONTOURS THAT IS MUCH DIFFERENT TO THE REST
    chosen_contours = remove_abnormal_contours(chosen_contours)

    # DRAW REMOVED CONTOURS
    white_image = 255 - np.zeros_like(gray)
    image_with_chosen_contours = cv2.drawContours(white_image, chosen_contours, contourIdx=-1, color=(0,0,255), thickness=1)
    # show_image(image_with_chosen_contours, name="contours")

    # GET THE IMAGE WITH CONTOURS AS COMPONENTS
    labels = np.zeros_like(gray) # dark image
    ret = 1
    for contour in chosen_contours:
        labels = cv2.drawContours(labels, [contour], contourIdx=-1, color=(ret, ret, ret), thickness=cv2.FILLED)
        ret += 1
    component_list(gray, ret, labels, chromosome_type=chromosome_type)

    return chosen_contours, ret, labels


def create_mask_contour(contours, shape):
    mask_contour = np.zeros(shape)
    mask_contour = mask_contour.astype('int')
    for id, contour in enumerate(contours):
        for point in contour:
            # X: point[0][1], Y: point[0][0]
            x = point[0][1]
            y = point[0][0]
            mask_contour[x][y] = id + 1
    return mask_contour


def unsharp_masking(image, radius=5, mask_weight=10):
    gaussian = cv2.GaussianBlur(image, (radius, radius), sigmaX=1)
    unsharp_image = cv2.addWeighted(image, 1 + mask_weight, gaussian, - mask_weight, gamma=0)
    return unsharp_image


# def sharpen_edges(image, kernel_size=(5, 5)):
#     kernel_sharpening = np.array([[-1,-1,-1], 
#                                   [-1, 9,-1],
#                                   [-1,-1,-1]])
#     sharpened = cv2.filter2D(image, -1, kernel_sharpening)
#     return sharpened

'''
This function is to remove contours that is too large or too small compared to the rest.
'''
def remove_abnormal_contours(contours, acceptance_large_ratio=10, acceptance_small_ratio=10):
    num_contour = len(contours)
    areas = np.asarray([cv2.contourArea(contour) for contour in contours])
    total_area = np.sum(areas)
    remaining_areas = total_area - areas
    average_remaining_areas = remaining_areas / (num_contour - 1)
    chosen_contours = list()
    for i in range(num_contour):
        if (areas[i] <= acceptance_large_ratio * average_remaining_areas[i]) and (areas[i] * acceptance_small_ratio >= average_remaining_areas[i]):
            chosen_contours.append(contours[i])
    return chosen_contours


# def colorize_contours(image, contours):
#     image_with_chosen_contours = cv2.drawContours(white_image, chosen_contours, contourIdx=-1, color=2, thickness=1)
#     return img

def watershed(image, contours):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    component_count = 1
    markers = np.zeros_like(gray)

    for contour in contours:
        markers = cv2.drawContours(markers, contour, contourIdx=-1, color=component_count, thickness=1)
        component_count += 1

    # markers = np.uint8(markers)
    markers = markers.astype('int')
    show_image(markers)

    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255,0,0]
    show_image(image)
    return image

'''
    Opening-by-reconstruction = Erosion + Morphological reconstruction
'''
def opening_by_reconstruction(image):
    # Erosion
    se = cv2.getStructuringElement(cv2.MORPH_ERODE, (20, 20)) # structure element
    Ie = cv2.erode(image, se, iterations=1)
    show_image(Ie)

    # Morphological reconstruction = iteratively dilation
    Iobr = reconstruction(Ie, image, method='dilation')
    return Iobr


def histogram_equalization(image, type="clahe"):
    if type == "normal":
        result = cv2.equalizeHist(image)
        return result
    if type == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        result = clahe.apply(image)
        return result

    return None


def component_list(gray, ret, labels, component_size=200, chromosome_type=None):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # ret, labels, points, area = connected_component_with_stats(binary)
    height, width = gray.shape
    ret, labels, points, area = connected_component_with_stats(ret, labels)

    for label in range(1, ret):
        # can replace by bounding rectangle
        x_points = [x for (x, y) in points[label]]
        y_points = [y for (x, y) in points[label]]
        x_min, x_max, y_min, y_max = min(x_points), max(x_points), min(y_points), max(y_points)

        white_image = 255 - np.zeros_like(gray)
        white_image[labels == label] = gray[labels == label]
        left_x = int((x_max + x_min - component_size) / 2)
        right_x = int((x_max + x_min + component_size) / 2)
        left_y = int((y_max + y_min - component_size) / 2)
        right_y = int((y_max + y_min + component_size) / 2)
        if (left_x >= 0 and right_x < height and left_y >= 0 and right_y < width):
            chromosome_image = sub_matrix(white_image, left_x, right_x, left_y, right_y)
            if chromosome_type is not None:
                path = image_dir + chromosome_type + '_chromosomes/' + chromosome_type + '_' + str(label) + '.BMP'
                print("Writing to " + path)
                cv2.imwrite(path, chromosome_image)
            # show_image(chromosome_image)

def connected_component_with_stats(ret, labels):    
    # ret, labels = cv2.connectedComponents(binary)
    points = dict()
    area = dict()
    for i in range(0, ret):
        points[i] = list()
        area[i] = 0
    height, width = labels.shape
    for i in range(height):
        for j in range(width):
            points[labels[i][j]].append((i, j))
            area[labels[i][j]] += 1

    return ret, labels, points, area

def sub_matrix(matrix, left_x, right_x, left_y, right_y):
    return matrix[np.ix_(range(left_x, right_x + 1), range(left_y, right_y + 1))]