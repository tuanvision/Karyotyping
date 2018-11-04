import cv2
from os.path import dirname, abspath
import numpy as np
from matplotlib import pyplot as plt
from skeleton import skeleton_3d, skeleton_normal
from threshold import partial_otsu_threshold, variable_threshold
from util import show_image


image_dir = dirname(dirname(abspath("X"))) + "/data/"

for i in range(32):
	image = cv2.imread(image_dir + "/giemsa_chromosomes/giemsa_" + str(i + 1) + ".BMP")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	thresh = partial_otsu_threshold(gray, minval=1, maxval=255, dark_background=False)
	cv2.imwrite(image_dir + "/giemsa_chromosomes/giemsa_" + str(i + 1) + "_r1_thresh.BMP", thresh)

	binary = thresh.copy()
	binary[binary == 255] = 1
	skeleton = skeleton_3d(binary)
	skeleton[skeleton == 1] = 255
	cv2.imwrite(image_dir + "/giemsa_chromosomes/giemsa_" + str(i + 1) + "_r2_skeleton.BMP", skeleton)


