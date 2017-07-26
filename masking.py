import cv2
import numpy as np
import os
import csv
import math
import time
from numba import jit

@jit
def crop_circle(image, radius, centerX, centerY):
	recX = 2*centerX
	recY = 2*centerY		
	start_time = time.time()	
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if int(math.pow((i - centerX),2)/0.6 + math.pow((j - centerY),2)) > math.pow(radius,2):
				image[i,j] = 0
	print("--- %s seconds ---" % (time.time() - start_time))
	return image





image_mask = cv2.imread("sss.tif");
b,green_fundus,r = cv2.split(image_mask)
dim = (800,615)
fundus1 = cv2.resize(green_fundus, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("dfddfsse",fundus1)
print(fundus1.shape)
centerY = 400
centerX = 307
radius = 370
qimage = crop_circle(fundus1,radius,centerX,centerY)
cv2.imshow("dfddfe",qimage)
cv2.waitKey()

