import numpy as np
import cv2


file_image = '../images/5180104844.020.A.BMP'
img = cv2.imread(file_image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret1, th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Coutours
im2, contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# image_with_contours = cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
print("Len contour: ", len(contours))
# cv2.imshow("contour:", image_with_contours)
# img2, con =  draw_contours(gray)

# rectangles
for contour in contours:
    size = cv2.contourArea(contour)
    ret_x, ret_y, ret_w, ret_h = cv2.boundingRect(contour)
    cv2.rectangle(img.copy(), (ret_x, ret_y),
                  (ret_x + ret_w,
                   ret_y + ret_h),
                  (0, 255, 0), 3)

# cv2.imshow("contour rec:", image_with_contours)

image_with_contours = cv2.drawContours(img.copy(), contours, 6, (0, 0, 255), 0)
cv2.imshow("contour:", image_with_contours)

cv2.waitKey()  # Wait for key stroke

for i,c in enumerate(contours):
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    box = cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    cropped = img[y: y+h, x: x+w]
    # cv2.imshow("Show Boxes", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("../imgen/blobby"+str(i)+".png", cropped)


cv2.imshow("Show Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()