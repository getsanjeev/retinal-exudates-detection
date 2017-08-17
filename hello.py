import cv2
import numpy as np
import os
import csv
import time
import math
from numba import jit


if __name__ == "__main__":
	pathFolder1 = "/home/sherlock/Internship@iit/exudate-detection/diaretdb_hardexudates/"
	pathFolder2 = "/home/sherlock/Internship@iit/exudate-detection/diaretdb_resized/"
	filesArray1 = [x for x in os.listdir(pathFolder1) if os.path.isfile(os.path.join(pathFolder1,x))]
	filesArray2 = [x for x in os.listdir(pathFolder2) if os.path.isfile(os.path.join(pathFolder2,x))]
	DestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/diaretdb1-label/"	

	if not os.path.exists(DestinationFolder):
		os.mkdir(DestinationFolder)	

	for file_name in filesArray1:
		#print(pathFolder1+'/'+file_name)
		file_name_no_extension = os.path.splitext(file_name)[0]
		print(file_name_no_extension)
		fundus1 = cv2.imread(pathFolder1+'/'+file_name)
		fundus2 = cv2.cvtColor(fundus1,cv2.COLOR_BGR2GRAY)
		dim = (800,615)
		candidate_label = cv2.resize(fundus2,dim)
		threshold = np.amax(candidate_label)
		if threshold-10 < 0:
			threshold = threshold+50
		else:
			threshold = threshold -10
		ret,bin_label = cv2.threshold(candidate_label,threshold,255,cv2.THRESH_BINARY)
		#cv2.imwrite(DestinationFolder+file_name_no_extension+"label.jpg",bin_label)

		original_fundus = cv2.imread(pathFolder2+'/'+file_name_no_extension+'_resized.jpg')
		b,g,r = cv2.split(original_fundus)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		enhanced_original_fundus = clahe.apply(g)
		print(enhanced_original_fundus.shape,bin_label.shape)		
		candidate_label = cv2.bitwise_and(enhanced_original_fundus,bin_label)
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_candidate_label.jpg",candidate_label)

		#candidate_label = cv2.bitwise_and(bin_label,)
				
		
