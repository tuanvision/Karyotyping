import cv2
import numpy as np
from os.path import dirname, abspath
from util import show_image, show_contour_image
from skeleton import skeleton_3d
from util import read_list, write_list
from threshold import partial_otsu_threshold
from util import match_shape, sift_matching

image_dir = dirname(dirname(abspath("X"))) + "/data/"
chromosome_contours_file = image_dir + "/chromosome_basic/chromosome_contours.npy"
chromosome_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "x", "y"]
num_chromosome = len(chromosome_name)

def compute_chromosome_contours():
    chromosome_contours = list()
    for name in chromosome_name:
        gray = cv2.imread(image_dir + "/chromosome_basic/chromosome_" + name + ".PNG", 0)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = thresh.copy()
        binary[binary == 255] = 1
        skeleton = skeleton_3d(binary)
        skeleton[skeleton == 1] = 255
        _, contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("Showing " + name)
        show_contour_image([contours[0]], gray.shape)
        chromosome_contours.append(contours[0])
    
    np.save(chromosome_contours_file, chromosome_contours)

def compute_chromosome_skeleton():
    chromosome_skeletons = list()
    for name in chromosome_name:
        gray = cv2.imread(image_dir + "/chromosome_basic/chromosome_" + name + ".PNG", 0)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = thresh.copy()
        binary[binary == 255] = 1
        skeleton = skeleton_3d(binary)
        skeleton[skeleton == 1] = 255
        cv2.imwrite(image_dir + "/chromosome_basic/chromosome_" + name + "_skeleton.PNG", skeleton)

def get_chromosome_contours():
    chromosome_contours = np.load(chromosome_contours_file)
    return chromosome_contours

def is_chromosome(gray, threshold=0):
    thresh = partial_otsu_threshold(gray, minval=1, maxval=255, dark_background=False)
    binary = thresh.copy()
    binary[binary == 255] = 1
    skeleton = skeleton_3d(binary)
    skeleton[skeleton == 1] = 255
    show_image(skeleton)

    for name in chromosome_name:
    	database_skeleton = cv2.imread(image_dir + "/chromosome_basic/chromosome_" + name + "_skeleton.PNG", 0)
    	show_image(database_skeleton)
    	matching_value = match_shape(skeleton, database_skeleton)
    	print("Matching " + name + " : " + str(matching_value))

    return True

# is_chromosome(cv2.imread(image_dir + "house.jpg", 0))

gray1 = cv2.imread(image_dir + "fish.PNG", 0)
thresh = partial_otsu_threshold(gray1, minval=1, maxval=255, dark_background=False)
binary = thresh.copy()
binary[binary == 255] = 1
skeleton = skeleton_3d(binary)
skeleton[skeleton == 1] = 255
gray1 = skeleton.copy()

gray2 = cv2.imread(image_dir + "/chromosome_basic/chromosome_1_skeleton.PNG", 0)
sift_matching(gray1, gray2)