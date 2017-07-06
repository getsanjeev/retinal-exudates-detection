import cv2
import numpy as np
import os
import csv
import time
from numba.decorators import jit, autojit

def crop_circle(image, radius, centerX, centerY):
	recX = 2*centerX
	recY = 2*centerY
	cropped_image = np.zeros((recX,recY), dtype="uint8") * 255
	start_time = time.time()
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if(i ==j):
				image[i,j] = 0
	print("--- %s seconds ---" % (time.time() - start_time))
	return image





image_mask = cv2.imread("sss.tif");
b,green_fundus,r = cv2.split(image_mask)
dim = (800,615)
fundus1 = cv2.resize(green_fundus, dim, interpolation = cv2.INTER_AREA)
centerX = 400
centerY = 307
radius = 290
#pairwise_numba = autojit(crop_circle)
qimage = crop_circle(fundus1,radius,centerY,centerY)
cv2.imshow("dfddfe",qimage)
cv2.waitKey()

