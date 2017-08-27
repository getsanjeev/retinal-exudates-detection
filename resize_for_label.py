import cv2
import numpy as np
import os
import csv
import time
import math
from numba import jit


if __name__ == "__main__":
	pathFolder = "/home/sherlock/Internship@iit/exudate-detection/diaretdb1/"
	filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
	bvDestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/diaretdb_resized/"	
	if not os.path.exists(bvDestinationFolder):
		os.mkdir(bvDestinationFolder)	

	for file_name in filesArray:
		print(pathFolder+'/'+file_name)
		file_name_no_extension = os.path.splitext(file_name)[0]
		fundus = cv2.imread(pathFolder+'/'+file_name)
		dim = (800,615)
		fundus1 = cv2.resize(fundus, dim, interpolation = cv2.INTER_AREA)		
		cv2.imwrite(bvDestinationFolder+file_name_no_extension+"_resized.bmp",fundus1)				
