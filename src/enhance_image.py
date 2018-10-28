# Standard import
#  conda install --channel https://conda.anaconda.org/menpo opencv3
import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
file_image = '../images/100864.013.A.BMP'


def show_image(image, name='image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread(file_image)

# Step 1: Convert image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray, 'gray')

# Otsu's method without noise filtering
ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imwrite("data/giemsa_raw_thresholded.BMP", th1)
# plt.imshow(th1, 'gray')

# Coutours
im2, contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)

# print(image_with_contours[0][0])
# show_image(image_with_contours)
# mask for image

#
mask_contour = np.zeros(image_with_contours.shape)
num_contour = len(contours)

image_enhance_contour = image_with_contours
# print(mask_countour.shape)
# print(image_with_contours)

# check every window with windowSize, stepSize: stride
def sliding_window(image, stride, window_size):
    # slide a window across the image
    print("Shape: ", image.shape[0], image.shape[1])
    for y in range(0, image.shape[0], stride):
        for x in range(0, image.shape[1], stride):
            # yield the current window
            # print(x, y, x + window_size[0], y + window_size[1])
            # arr = np.array([x, y, x + windowSize[0], y + windowSize[1] ])
            arr_window = mask_contour[x: x + window_size[0], y: y + window_size[1]]
            num_con_window = np.count_nonzero(np.unique(arr_window))
            # print("Number contour in this window:", num_con_window)
            if num_con_window  <= 1:
                image_enhance_contour[ x: x + window_size[0], y: y + window_size[1]] = 0
            # yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


print("Len contours: ", len(contours));
cnt = 0
for id, contour in enumerate(contours):
    # print(id, len(c), image_with_contours[c[0][0][0]][c[0][0][1] ] )
    #     for j in c:
    #         # X: j[0][1], Y: j[0][0]
    #         print(image_with_contours[j[0][1]][j[0][0]])
    for point in contour:
        # X: point[0][1], Y: point[0][0]
        x = point[0][1]
        y = point[0][0]
        mask_contour[x][y] = id + 1
        # print(len(c[0]), c[0][0], type(c[0][0]))
    cnt += len(contour)

# print(cnt)
print(image_with_contours.shape)

sliding_window(image_with_contours, 40, (70, 70))
# show_image(image_enhance_contour, "image_enhance_contour")
cv2.imwrite("../images/image_enhance_contour.BMP", image_enhance_contour)

