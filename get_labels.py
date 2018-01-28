import cv2
import numpy as np
import os
import csv
import time
import math
from numba import jit


@jit
def get_roi_mean(image):	
	i = 0
	j = 0
	sum = 0
	count = 0
	timii = time.time()
	print("in the fn")
	while i< image.shape[0]:
		j = 0
		while j < image.shape[1]:			
			if(image[i,j] != 0):
				sum = sum + image[i,j]
				count = count + 1
			j = j +1
		i = i +1	
	print(time.time() - timii)
	if count ==0:
		count = 1
	print("count",count)
	return sum/count



if __name__ == "__main__":
	pathFolder1 = "/home/sherlock/Internship@iit/exudate-detection/diaretdb_hardexudates/"
	pathFolder2 = "/home/sherlock/Internship@iit/exudate-detection/diaretdb_resized/"
	filesArray1 = [x for x in os.listdir(pathFolder1) if os.path.isfile(os.path.join(pathFolder1,x))]
	#filesArray2 = [x for x in os.listdir(pathFolder2) if os.path.isfile(os.path.join(pathFolder2,x))]
	DestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/diaretdb1-label111/"	

	if not os.path.exists(DestinationFolder):
		os.mkdir(DestinationFolder)	

	for file_name in filesArray1:
		
		file_name_no_extension = os.path.splitext(file_name)[0]
		print(file_name_no_extension)
		fundus1 = cv2.imread(pathFolder1+'/'+file_name)
		b,fundus2,r = cv2.split(fundus1)		
		dim = (800,615)
		candidate_label = cv2.resize(fundus2,dim)
		threshold = np.amax(candidate_label)
		if threshold-10 < 0:
			threshold = threshold+50
		else:
			threshold = threshold -10

		ret,bin_label = cv2.threshold(candidate_label,threshold,255,cv2.THRESH_BINARY)
		#cv2.imwrite(DestinationFolder+file_name_no_extension+"label.jpg",bin_label)
		original_fundus = cv2.imread(pathFolder2+'/'+file_name_no_extension+'_resized.bmp')
		b,g,r = cv2.split(original_fundus)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		enhanced_original_fundus = clahe.apply(g)
		
		candidate_label = cv2.bitwise_and(enhanced_original_fundus,bin_label)		
		ret,fin_label = cv2.threshold(candidate_label,get_roi_mean(candidate_label)+10,255,cv2.THRESH_BINARY)
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_final_label.bmp",fin_label)
		print(fin_label.shape,"www")
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_candidate_label.bmp",candidate_label)

		#candidate_label = cv2.bitwise_and(bin_label,)
				
		
