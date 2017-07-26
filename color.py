import cv2
import numpy as np
import os
import csv


def line_of_symmetry(image):
	image_v = image.copy()
	prev = 0
	for i in range(image_v.shape[0]):
		x, y = (image_v).nonzero()
		count = x.size()
		if count > prev:
			prev = count			
		i = i + 35
	print(prev)
	#x, y = imag_v.nonzero()
	#image[x,y] = 66
	return image


def remove_OD(image):
	image_x = image.copy()
	b,green_fundus,red_fundus = cv2.split(resized_fundus)
	# circles = cv2.HoughCircles(green_fundus,cv2.HOUGH_GRADIENT,1,50,
 #                            param1=40,param2=30,minRadius=0,maxRadius=0)
	# cimg = green_fundus
	# for i in circles[0,:]:
	# 	cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
	# 	cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
	# 	cv2.imshow('detected circles',cimg)
	ret,g_channel = cv2.threshold(green_fundus,(np.amax(green_fundus)+np.mean(green_fundus))/2,255,cv2.THRESH_BINARY)
	ret,r_channel = cv2.threshold(red_fundus,(np.amax(red_fundus)+np.mean(red_fundus))/2,255,cv2.THRESH_BINARY)	
	if(g_channel == r_channel).nonzero():
		image_x = g_channel
	else:
		image_x = 0
	# for i in range(image_x.shape[0]):
	# 	for j in range(image_x.shape[1]):
	# 		if(g_channel[i,j] == r_channel[i,j]):
	# 			image_x[i,j] = g_channel[i,j]
	# 		else:
	# 			image_x[i,j] = 0
	return image_x







if __name__ == "__main__":
    pathFolder = "/home/sherlock/Internship@iit/exudate-detection/DRIVE/"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    destinationFolder = "/home/sherlock/Internship@iit/exudate-detection/DRIVE-exudates-color/"

    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)

    for file_name in filesArray:
    	print(pathFolder+'/'+file_name)
    	file_name_no_extension = os.path.splitext(file_name)[0]
    	fundus = cv2.imread(pathFolder+'/'+file_name)
    	dim = (800,615)
    	resized_fundus = cv2.resize(fundus, dim, interpolation = cv2.INTER_AREA)    	
    	eroded_image = remove_OD(resized_fundus)
    	cv2.imwrite(destinationFolder+file_name_no_extension+"_exudates.jpg",eroded_image)



    
