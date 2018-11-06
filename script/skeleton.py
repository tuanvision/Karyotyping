# -*- coding: utf-8 -*-
import numpy as np
import cv2
from skimage.morphology import reconstruction, remove_small_objects, remove_small_holes
from os.path import dirname, abspath
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis
from preprocessing import show_image
from skimage.morphology import skeletonize, skeletonize_3d, thin
from util import show_image, show_contour_image, isclose


def skeleton_normal(binary):
    return skeletonize(binary) * 1


def skeleton_3d(binary):
    return skeletonize_3d(binary) * 1


def topology_skeleton(gray, return_skel_points=False):
    print(gray.shape)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    # draw the contour
    white_image = 255 - np.zeros_like(gray)
    contour_image = cv2.drawContours(white_image, [contour], contourIdx=-1, color=(0,0,0), thickness=1)
    show_image(contour_image)

    # remove the contour part from image
    binary_without_boundary = binary.copy()
    binary_without_boundary[contour_image == 0] = 0
    show_image(binary)
    show_image(binary_without_boundary)

    
    # perform topology skeleton
    skel, distance = medial_axis(binary_without_boundary, return_distance=True) 
    show_image(skel)

    # get list of points on skeleton
    skel_points = list()
    height, width = gray.shape
    for i in range(height):
        for j in range(width):
            if skel[i, j] == 1:
                skel_points.append([[i, j], distance[i, j]])

    # apply discrete curve evolution to obtain partition points, then get segments between each pairs of partition points
    partition_points = discrete_curve_evolution(contour, binary)
    print(len(contour))
    print(len(partition_points))
    num_partition_points = len(partition_points)
    segments = list()
    current_idx = 0
    for i in range(num_partition_points):
        curr_point = partition_points[i]
        next_point = partition_points[(i + 1) % num_partition_points]
        segment = list()
        while not np.array_equal(contour[current_idx], next_point):
            segment.append(contour[current_idx][0])
            current_idx += 1
            if current_idx >= len(contour):
                break
        segments.append(segment)
    print(len(segments))

    # remove unimportant skeleton points
    pruned_skel_points = list()

    # duyệt qua từng điểm thuộc skeleton
    for [skel_point, dist] in skel_points: # dist là khoảng cách gần nhất từ điểm tới biên
        already_in_other_segment = False # tham số kiểm tra xem đã có một generating point thuộc một segment hay chưa
        point_pass = False # tham số để xem điểm này có được chọn hay không
        for segment in segments: # duyệt qua từng segment
            num_point = len(segment)
            for i in range(num_point - 1): # duyệt qua từng đường nối hai điểm liên tiếp trong segment
                curr_point = segment[i]
                next_point = segment[i + 1]
                if isclose(dist, point_to_line_distance(curr_point, next_point, skel_point), rel_tol=1e-02): # nếu khoảng cách từ skeleton point tới đường nối = dist
                    if already_in_other_segment: # nếu đã có generating point thuộc một segment khác
                        point_pass = True # nghĩa là điểm này có generating points thuộc các segment khác nhau, do đó được chọn
                        break
                    else:
                        already_in_other_segment = True # ám chỉ rằng từ giờ đã có generating point thuộc 1 segment nào đó
                        break

            if point_pass:
                break

        if point_pass:
            pruned_skel_points.append(skel_point)

    print(len(pruned_skel_points))
    pruned_skel = np.zeros_like(gray)
    for [x, y] in pruned_skel_points:
        pruned_skel[x, y] = 255
    show_image(pruned_skel)

    if return_skel_points:
        return pruned_skel, pruned_skel_points
    else:
        return pruned_skel


def discrete_curve_evolution(contour, binary=None, num_iter=1000):
    polygon = contour

    for i in range(num_iter):
    	print(i)
        # minimum relevance value 
        min_value = 1e9

        # current points (contour)
        points = polygon.copy()

        # length of contour
        total_length = cv2.arcLength(points, closed=True)

        # the number of points in contour
        num_point = len(points)

        # list of upcoming chosen points 
        polygon = list()

        # relevence values corresponding to each point
        relevance_values = list()

        # compute relevance values
        for j in range(num_point):            
            last_point = points[(j - 1) % num_point][0]
            curr_point = points[j][0]
            next_point = points[(j + 1) % num_point][0]
            value = relevance_value([last_point, curr_point, next_point], total_length)
            min_value = min(min_value, value)
            relevance_values.append(value)

        # eliminate points with minimum relevance value
        for j in range(num_point):
            # skip point if its value is minimum
            if isclose(relevance_values[j], min_value):
                continue

            # else add it to the list
            polygon.append(points[j])

        if len(polygon) < 10:
        	break

        if (binary is not None) and ((i + 1) % 50 == 0 or len(polygon) < 15):
        	polygon = np.asarray(polygon)
        	show_contour_image([polygon], binary.shape)
        	continue

        polygon = np.asarray(polygon)

    return polygon


def relevance_value(points, total_length):
    last_point, curr_point, next_point = points
    angle = np.pi - angle_between(next_point - curr_point, last_point - curr_point)
    l1 = point_distance(next_point, curr_point) / total_length
    l2 = point_distance(last_point, curr_point) / total_length
    value = angle * l1 * l2 / (l1 + l2)
    return value

def point_distance(x, y, type='euclidean'):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.linalg.norm(x - y)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def is_sublist(l1, l2):
    return set(l1).issubset(set(l2))

def point_in_segment(point, segment):
    for seg_point in segment:
        if np.array_equal(point, seg_point):
            return True
    return False

def all_points_in_segment(points, segment):
    for point in points:
        if not point_in_segment(point, segment):
            return False
    return True

def list_contains(l, e):
    x_1, y_1 = e
    for x_2, y_2 in l:
        if x_1 == x_2 and y_1 == y_2:
            return True

    return False

def point_to_line_distance(p1, p2, p3):
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

def test_skeleton():
	square = np.zeros((13, 13), dtype=np.uint8)
	square[1:-1, 2:-2] = 1
	# square[2:-2, 3:-3] = 0
	square[1, 2] = 0
	print(square)
	skel, distance = medial_axis(square, return_distance=True)
	print(skel * 1)
	print(np.around(distance, decimals=2))
	square[1, 2] = square[1, -3] = square[-2, 2] = square[-2, -3] = 0
	print(square)
	discrete_curve_evolution(square)


# image_dir = dirname(dirname(abspath("X"))) + "/data/"	
# gray = cv2.imread(image_dir + "fish.PNG", 0)
# show_image(gray)
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# show_image(thresh)
# _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# show_contour_image([contours[0]], gray.shape)
# print(discrete_curve_evolution(contours[0], gray))
