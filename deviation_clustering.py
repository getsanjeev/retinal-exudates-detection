import numpy as np
import cv2
from numba import jit
import os
from matplotlib import pyplot as plt
import math



@jit
def deviation_from_mean(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_output = clahe.apply(image)
	print(clahe_output)
	result = clahe_output.copy()
	result = result.astype('int')
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			sub_image = clahe_output[i:i+5,j:j+5]
			mean = np.mean(sub_image)
			sub_image = sub_image - mean
			result[i:i+5,j:j+5] = sub_image
			j = j+5
		i = i+5
	return result



if __name__ == "__main__":
	pathFolder = "/home/sherlock/Internship@iit/exudate-detection/testing/"
	filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
	DestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/testing-result-kmeans-deviation/"
	
	if not os.path.exists(DestinationFolder):
		os.mkdir(DestinationFolder)	
	for file_name in filesArray:
		print(pathFolder+'/'+file_name)
		file_name_no_extension = os.path.splitext(file_name)[0]
		fundus = cv2.imread(pathFolder+'/'+file_name)
		fundus = cv2.resize(fundus,(800,615))
		fundus_mask = cv2.imread("fmask.tif")
		fundus_mask = cv2.resize(fundus_mask,(800,615))

		f1 = cv2.bitwise_and(fundus[:,:,0],fundus_mask[:,:,0])
		f2 = cv2.bitwise_and(fundus[:,:,1],fundus_mask[:,:,1])
		f3 = cv2.bitwise_and(fundus[:,:,2],fundus_mask[:,:,2])
		fundus_dash = cv2.merge((f1,f2,f3))

		b,g,r = cv2.split(fundus_dash)
		
		gray_scale = cv2.cvtColor(fundus_dash,cv2.COLOR_BGR2GRAY)
		
		ff = deviation_from_mean(gray_scale)

		print((ff>0).sum(),"greater")
		print((ff==0).sum(),"equal")
		print((ff<0).sum(),"smaller")

		print(ff[300:310,400:410])
		ff = np.reshape(ff,(ff.size,1))
		
		Z = np.hstack((ff))
		Z = np.float32(Z)

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
		ret,label,center=cv2.kmeans(Z,5,None,criteria,50,cv2.KMEANS_RANDOM_CENTERS)
		print(center)

		u, indices, counts = np.unique(label, return_index=True, return_counts=True)
		print(u,"unique labels in Clustered image")
		print(counts,"count of labels in Clustered image")
		

		center_t = [(t[0]) for t in center]
		print(center_t,"centre relevant to me")
		print(max(center_t),"max of center")
	

		green = [0,255,0]
		blue = [255,0,0]
		red = [0,0,255]
		white = [255,255,255]
		black = [0,0,0]
		pink = [220,30,210]
		sky = [30,240,230]
		yellow = [230,230,30]

		color = [white,black,red,green,blue,pink]
		color = np.array(color,np.uint8)
		label = np.reshape(label, gray_scale.shape)


		test = label.copy()				
		test[test!=u[center_t.index(max(center_t))]] = -1
		test[test==u[center_t.index(max(center_t))]] = 255
		test[test==-1] = 0

		y = color[label]
		y = np.uint8(y)
						
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_result_exudates_kmeans.bmp",y)				
				
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_final_candidates.bmp",test)

		print("-----------x-------DONE-------x----------")
