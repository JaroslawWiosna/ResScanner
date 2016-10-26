import numpy as np
import cv2
from matplotlib import pyplot as plt

def printIMG(image):
    cv2.imshow("Image",image)
    cv2.waitKey()

rez = cv2.imread("img/rez.jpg")

blue_lower = np.array([255, 65, 65])
blue_upper = np.array([200, 0, 0])

red_lower = np.array([65, 65, 255])
red_upper = np.array([0, 0, 200])

green_lower = np.array([65, 255, 65])
green_upper = np.array([0, 200, 0])

white_lower = np.array([255, 255, 255])
white_upper = np.array([200, 200, 200])

black_lower = np.array([65, 65, 65])
black_upper = np.array([0, 0, 0])

upper = red_upper
lower = red_lower

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
