# def point_hierarchy_multiple_curve(binary):
#     # dictionary = {id: point_info, ...}
#     # point_info = [point, [neighbors]]
#     dictionary = dict()
#     height, width = binary.shape
#     idx = 0
#     for (x, y) in np.nindex(height, width):
#         if binary[x][y] == 1:
#             neighbors = list()
#             for i in [x - 1, x, x + 1]:
#                 for j in [y - 1, y, y + 1]:
#                     if i < 0 or j < 0:
#                         continue
#                     if i >= width or j >= height:
#                         continue
#                     if binary[i][j] == 1:
#                         neighbors.append((i, j))
#         dictionary[idx] = [(x, y), neighbors]
#         idx += 1

#     return dictionary

# # OK
# class PointList:
#     def __init__(self, headval=None):
#         self.headval = headval
#         self.currval = self.headval
#         self.size = None

#     def add_point(self, point):
#         if self.headval is None:
#             self.headval = point
#             self.currval = self.headval
#             return
#         self.currval.nextval = point
#         point.lastval = self.currval
#         self.currval = point

#     def draw(self, shape):
#         black_image = np.zeros(shape)
#         currval = self.headval
#         while True:
#             curr_x, curr_y = currval.dataval
#             next_x, next_y = currval.nextval.dataval
#             cv2.line(img=black_image, pt1=(curr_x, curr_y), pt2=(next_x, next_y), color=(255, 255, 255), thickness=1)
#             currval = currval.nextval
#             if currval is self.headval:
#                 break
#         return black_image

#     def listprint(self):
#         data_list = list()
#         printval = self.headval
#         size = 0
#         count = 0
#         while True:
#             data_list.append(printval.dataval)
#             printval = printval.nextval
#             size += 1
#             count += 1
#             if count == 50:
#                 break
#             if printval is self.headval:
#                 break

#         self.size = size
#         print(self.size)
#         print(data_list)
#         return data_list

#     def get_size(self):
#         if self.headval is None:
#             return 0
#         if self.size is not None:
#             return self.size

#         size = 0
#         currval = self.headval
#         while True:
#             size += 1
#             currval = currval.nextval
#             if currval is self.headval:
#                 break

#         self.size = size
#         return size


#     def total_length(self):
#         length = 0
#         currval = self.headval
#         while True:
#             length += distance(currval.dataval, currval.nextval.dataval)
#             currval = currval.nextval
#             if currval.equals(self.headval):
#                 break

#         return length

# class Point:
#     # def __init__(self, value=None):
#     #     self.dataval = value
#     #     self.lastval = None
#     #     self.nextval = None
#     #     self.relavance = 1e9

#     # def equals(self, p2):
#     #     x_1, y_1 = self.dataval
#     #     x_2, y_2 = p2.dataval
#     #     if x_1 == x_2 and y_1 == y_2:
#     #         return True
#     #     return False

#     # def touching_neighbors(self):
#     #     x, y = self.dataval
#     #     return [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]

#     # def diagonal_neighbors(self):
#     #     x, y = self.dataval
#     #     return [[x - 1, y - 1], [x + 1, y - 1], [x - 1, y + 1], [x + 1, y + 1]]

#     # def neighbors(self):
#     #     x, y = self.dataval
#     #     return [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1], [x - 1, y - 1], [x + 1, y - 1], [x - 1, y + 1], [x + 1, y + 1]]

# # OK
# def point_hierarchy(binary):
#     seed_point = None
#     height, width = binary.shape
    
#     # find seed point
#     for (x, y) in np.ndindex(height, width):
#         if binary[x][y] == 1:
#             seed_point = [x, y]
#             break
#     if seed_point is None:
#         raise("No seed point!")

#     # contour traversal
#     points = PointList(Point(seed_point))
#     last_point = None
#     next_point = None
#     curr_point = points.headval
#     while True:
#         print(1)
#         neighbors = curr_point.neighbors()
#         active_neighbors = list()
#         for x_nb, y_nb in neighbors:
#             if binary[x_nb, y_nb] == 1:
#                 active_neighbors.append([x_nb, y_nb])

#         if last_point is None:
#             last_point = Point(active_neighbors[0])
    
#         last_x, last_y = last_point.dataval

#         for neighbor in active_neighbors:
#             if last_point.equals(Point(neighbor)):
#                 continue
#             if list_contains(last_point.touching_neighbors(), neighbor):                
#                 continue

#             next_point = Point(neighbor)
#             break

#         if next_point is None:
#             raise Exception(("No next point!"))

#         if next_point.equals(points.headval):
#             points.add_point(points.headval)
#             break

#         # print(last_point.dataval, curr_point.dataval, next_point.dataval)

#         points.add_point(next_point)
#         last_point = curr_point
#         curr_point = next_point

#     points.listprint()
        
#     return points, points.get_size()

    # points, num_point = point_hierarchy(binary)
    # polygon, num_vertice = None, None


    # for i in range(num_iter):
    #     min_value = 1e9
    #     polygon = PointList()

    #     # calculate relevance value for each points
    #     curr_point = points.headval
    #     while True:
    #         value = relevance_value(curr_point, total_length)
    #         curr_point.relevance = value
    #         min_value = min(min_value, value)
    #         curr_point = curr_point.nextval
    #         if curr_point.equals(points.headval):
    #             break

    #     # eliminate points with minimum relevance value
    #     curr_point = points.headval
    #     head_polygon = None
    #     while True:
    #         if isclose(curr_point.relevance, min_value):
    #             print(curr_point.dataval)
    #             curr_point = curr_point.nextval
    #             if curr_point.equals(points.headval):
    #                 break
    #             continue
    #         if polygon.get_size() == 0:
    #             head_polygon = curr_point
    #         polygon.add_point(curr_point)
    #         curr_point = curr_point.nextval
    #         if curr_point.equals(points.headval):
    #             polygon.add_point(head_polygon)
    #             break

    #     polygon.listprint()
    #     show_image(polygon.draw(binary.shape))
    #     points = polygon

    # return polygon



#     import numpy as np
# import cv2

# def background_removal(image, stride=20, window_size=(100, 100), num_contour_threshold=3, acceptance_ratio=2):
#     # read image
#     gray = cv2.imread(image)

#     # threshold by otsu's method
#     ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # get contours
#     img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     image_with_contours = cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)

#     # mask each contour by a number (id)
#     mask_contour = create_mask_contour(contours, shape=gray.shape)

#     # create counting dictionary of how many valid and invalid windows for each contour
#     num_contour = len(contours)
#     valid = {contour_id: 0 for contour_id in range(1, num_contour + 1)}
#     in_valid = {contour_id: 0 for contour_id in range(1, num_contour + 1)}

#     # slide window
#     for y in range(0, gray.shape[0], stride):
#         for x in range(0, gray.shape[1], stride):
#             # get window
#             arr_window = mask_contour[x: x + window_size[0], y: y + window_size[1]]

#             # get contours in the window 
#             unique, counts = np.unique(arr_window, return_counts=True)         

#             # update 2 dictionaries 
#             # a window is valid if the number of contours in that window is greater than a threshold, else invalid.
#             num_con_window = np.count_nonzero(unique)
#             if num_con_window > num_contour_threshold :
#                 for con_id in unique:
#                     if con_id == 0: 
#                         continue
#                     valid[con_id] = valid[con_id] + 1
#             else:
#                 for con_id in unique:
#                     if con_id == 0: 
#                         continue
#                     in_valid[con_id] = in_valid[con_id] + 1

#     # create list of contours to remove
#     # a contour will be removed if #valid/#invalid is lower than a ratio
#     removal_contours = list()
#     for i in range(1, num_contour + 1):
#         if valid[i] * acceptance_ratio < in_valid[i]:
#             new_contours.append(contours[i - 1])

#     # draw removed contours
#     white_image = 255 - np.zeros_like(gray)
#     image_with_removed_contours = cv2.drawContours(white_image, removal_contours, contourIdx=-1, color=2, thickness=1)
#     show_image(image_with_removed_contours)
#     return removal_contours


# def create_mask_contour(contours, shape):
#     mask_contour = np.zeros(shape)
#     mask_contour = mask_contour.astype('int')
#     for id, contour in enumerate(contours):
#         for point in contour:
#             # X: point[0][1], Y: point[0][0]
#             x = point[0][1]
#             y = point[0][0]
#             mask_contour[x][y] = id + 1
#     return mask_contour


# def show_image(image, name='image'):
#     cv2.imshow(name,image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
