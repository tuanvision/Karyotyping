import cv2
from preprocessing import background_removal, show_image, unsharp_masking, opening_by_reconstruction, grisan_local_threshold, adaptive_otsu_threshold
from os.path import dirname, abspath
import numpy as np
from matplotlib import pyplot as plt

image_dir = dirname(dirname(abspath("X"))) + "/data/"
img = cv2.imread(image_dir + "giemsa_raw.BMP")
# unsharp_img = unsharp_masking(filtered_img)
# show_image(img)
# show_image(filtered_img)
# show_image(unsharp_img)

# # unsharp_img = unsharp_masking(img)
# # show_image(img)
# # show_image(unsharp_img)

# contours = background_removal(img)
# image_with_contours = cv2.drawContours(img, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)
# show_image(image_with_contours)

show_image(img)
show_image(adaptive_otsu_threshold(img))

# img = cv2.imread(image_dir + "dapi_raw.BMP", 0)
# Iobr = opening_by_reconstruction(img)
# cv2.imwrite(image_dir + "dapi_result_2610_1010.BMP", Iobr)
# show_image(Iobr)


# cv2.imwrite(image_dir + "dapi_result_2610.BMP", image_with_contours)