import numpy as np
import cv2

def background_removal(image, stride=20, window_size=(100, 100), num_contour_threshold=3, acceptance_ratio=2):
    # read image
    gray = cv2.imread(image)

    # threshold by otsu's method
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # get contours
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)

    # mask each contour by a number (id)
    mask_contour = create_mask_contour(contours, shape=gray.shape)

    # create counting dictionary of how many valid and invalid windows for each contour
    num_contour = len(contours)
    valid = {contour_id: 0 for contour_id in range(1, num_contour + 1)}
    in_valid = {contour_id: 0 for contour_id in range(1, num_contour + 1)}

    # slide window
    for y in range(0, gray.shape[0], stride):
        for x in range(0, gray.shape[1], stride):
            # get window
            arr_window = mask_contour[x: x + window_size[0], y: y + window_size[1]]

            # get contours in the window 
            unique, counts = np.unique(arr_window, return_counts=True)         

            # update 2 dictionaries 
            # a window is valid if the number of contours in that window is greater than a threshold, else invalid.
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

    # create list of contours to remove
    # a contour will be removed if #valid/#invalid is lower than a ratio
    removal_contours = list()
    for i in range(1, num_contour + 1):
        if valid[i] * acceptance_ratio < in_valid[i]:
            new_contours.append(contours[i - 1])

    # draw removed contours
    white_image = 255 - np.zeros_like(gray)
    image_with_removed_contours = cv2.drawContours(white_image, removal_contours, contourIdx=-1, color=2, thickness=1)
    show_image(image_with_removed_contours)
    return removal_contours


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


def show_image(image, name='image'):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
