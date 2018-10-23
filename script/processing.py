import numpy as np
import cv2
from matplotlib import pyplot as plt

def show_image(image, name='image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def counting_in_matrix(matrix):
    counting_dictionary = dict()
    for row in matrix:
        for element in row:
            if element in counting_dictionary:
                counting_dictionary[element] += 1
            else:
                counting_dictionary[element] = 1
    print(counting_dictionary)
    return counting_dictionary

def clean_image_by_counting_contours(image, contours):
    num_contour = len(contours)
    window_size = 50
    contour_dict = dict()
    delete_contours = dict()
    for i in range(num_contour):
        points_set = set()
        for j in range(len(contours[i])):
            points_set.add(contours[i][j])
        contour_dict[i] = points_set
    height, width = image.shape
    for center_x in range(window_size, height - window_size + 1):
        for center_y in range(window_size, width - window_size + 1):
            for i in range(center_x - window_size, center_x + window_size + 1):
                for j in range(center_y - window_size, center_y + window_size + 1):
                    for id in range(num_contour):
                        if np.asarray([i, j]) in contour_dict[id]:
                            delete_contours[id] = 1
    print(delete_contours)
    return delete_contours

img = cv2.imread('data/giemsa_raw.BMP')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Otsu's method without noise filtering
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Contours
white_image = 255 - np.zeros_like(gray)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = cv2.drawContours(white_image, contours, contourIdx=-1, color=2, thickness=1)
clean_image_by_counting_contours(thresh, contours)
# cv2.imwrite("data/giemsa_contours.BMP", image_with_contours)
# show_image(image_with_contours)

