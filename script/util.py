import numpy as np
import cv2
from skimage.morphology import reconstruction, remove_small_objects, remove_small_holes
from os.path import dirname, abspath
from matplotlib import pyplot as plt
import json

matplotlib.use('TkAgg')


def match_shape(gray1, gray2):
	hu_moment_1 = cv2.HuMoments(cv2.moments(gray1)).flatten()
	hu_moment_2 = cv2.HuMoments(cv2.moments(gray2)).flatten()
	matching_value = np.sum(np.abs(hu_moment_1 - hu_moment_2))
	return matching_value

def sift_matching(gray1, gray2):
	sift = cv2.xfeatures2d.SIFT_create()
	
	kp1, des1 = sift.detectAndCompute(gray1, None)
	kp2, des2 = sift.detectAndCompute(gray2, None)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)

	good = []
	for m,n in matches:
	    if m.distance < 0.75 * n.distance:
	        good.append([m])
	
	matching_image = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, good, flags=2)
	show_image(matching_image)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def show_image(image, name='image', cmap='gray', shower='pyplot'):
    if shower == "pyplot":
        plt.imshow(image, cmap=cmap), plt.show()
        return

    if shower == "cv":
        cv2.imshow(name,image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


def show_contour_image(contours, shape):
    black_image = np.zeros(shape)
    contour_image = cv2.drawContours(black_image, contours, contourIdx=-1, color=(255,255,255), thickness=1)
    show_image(contour_image)
    return contour_image


def write_list(l, file_name):
    with open(file_name, 'w') as filehandle:
        json.dump(l, filehandle)


def read_list(file_name):
    with open(file_name, 'r') as filehandle:
        l = json.load(filehandle)
        return l