import cv2
from preprocessing import background_removal, show_image, sharpen_edges
from os.path import dirname, abspath
import numpy as np
from matplotlib import pyplot as plt

image_dir = dirname(dirname(abspath("X"))) + "/data/"
img = cv2.imread(image_dir + "dapi_raw.BMP")
filtered_img = cv2.imread(image_dir + "dapi_median_filtered.BMP")
contours = background_removal(filtered_img)
image_with_contours = cv2.drawContours(img, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)
show_image(image_with_contours)

# cv2.imwrite(image_dir + "dapi_result_2610.BMP", image_with_contours)