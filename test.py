import numpy as np
import cv2
from matplotlib import pyplot as plt

def printIMG(image):
    cv2.imshow("Image",image)
    cv2.waitKey()

rez = cv2.imread("img/rez.jpg")
upper = np.array([65, 65, 255])
lower = np.array([0, 0, 200])
mask = cv2.inRange(rez, lower, upper)
cnts, hierarchy = cv2.findContours(mask.copy(), \
                                   cv2.RETR_TREE, \
                                   cv2.CHAIN_APPROX_SIMPLE)

c = max(cnts, key=cv2.contourArea)
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.05 * peri, True)
cv2.drawContours(rez, [approx], -1, (0, 255, 0), 4)
printIMG(rez)
cv2.imwrite("img/redFound.jpg",rez)

cv2.destroyAllWindows()
