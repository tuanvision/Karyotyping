import numpy as np
import cv2
from skimage.morphology import reconstruction, remove_small_objects, remove_small_holes

def background_removal(image, stride=20, window_size=(100, 100), num_contour_threshold=3, acceptance_ratio=2):
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(gray, name='gray')

    # use median filter to remove noise
    gray = cv2.medianBlur(gray, 5)

    # # normalize histogram
    # gray = histogram_equalization(gray, type="clahe")
    # show_image(gray, name="he")

    # # perform unsharp masking
    # gray = unsharp_masking(gray)
    # show_image(gray, name="unsharp")

    # threshold by otsu's method
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
    show_image(thresh, name="thresh")

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
    chosen_contours = list()
    for i in range(1, num_contour + 1):
        if valid[i] * acceptance_ratio >= in_valid[i]:
            chosen_contours.append(contours[i - 1])

    # remove contours that is much different to the rest
    chosen_contours = remove_abnormal_contours(chosen_contours)

    # draw removed contours
    white_image = 255 - np.zeros_like(gray)
    image_with_chosen_contours = cv2.drawContours(white_image, chosen_contours, contourIdx=-1, color=2, thickness=1)
    show_image(image_with_chosen_contours, name="contours")
    return chosen_contours


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


def show_image(image, name='image'):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

# Opening-by-reconstruction = Erosion + Morphological reconstruction
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

# reference:  Automatic segmentation and disentangling of chromosomes in Q-band prometaphase images
def grisan_local_threshold(image, grid_size=350):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
def adaptive_otsu_threshold(image, grid_size=10):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = np.zeros_like(gray)
    height, width = gray.shape    
    h_upper_bound = int(height/grid_size)
    w_upper_bound = int(width/grid_size)
    for i in range(h_upper_bound):
        for j in range(w_upper_bound):
            threshold[max(0, i - 1) * grid_size : i * grid_size][max(0, j - 1) * grid_size : j * grid_size] = get_grid_threshold(gray, i, j, grid_size)
    return threshold

def get_grid_threshold_value(gray, i, j, grid_size):
    sub_gray = gray[max(0, i - 1) * grid_size : i * grid_size][max(0, j - 1) * grid_size : j * grid_size]
    ret, thresh = cv2.threshold(sub_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret

def get_grid_threshold(gray, i, j, grid_size):
    sub_gray = gray[max(0, i - 1) * grid_size : i * grid_size][max(0, j - 1) * grid_size : j * grid_size]
    ret, thresh = cv2.threshold(sub_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
