import cv2
import numpy as np
import os
import csv

def crop_circle(image, radius, centerX, centerY):
	recX = 2*centerX
	recY = 2*centerY
	cropped_image = np.zeros((recX,recY), dtype="uint8") * 255
	if (abs(centerX-image) <= radius).nonzero() and (abs(centerY-image) <= radius).nonzero():
		cropped_image = image
	cv2.imwrite("cropp.jpg",cropped_image)	





image_mask = cv2.imread("sss.tif");
b,green_fundus,r = cv2.split(image_mask)
dim = (800,615)
fundus1 = cv2.resize(green_fundus, dim, interpolation = cv2.INTER_AREA)
centerX = 400
centerY = 307
radius = 290
crop_circle(fundus1,radius,centerY,centerY)

