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
	b,g_channel,r_channel = cv2.split(resized_fundus)
	imagexx = g_channel.copy()

	#ret,g_channel = cv2.threshold(green_fundus,(np.amax(green_fundus)+np.mean(green_fundus))/2,255,cv2.THRESH_BINARY)
	#ret,r_channel = cv2.threshold(red_fundus,(np.amax(red_fundus)+np.mean(red_fundus))/2,255,cv2.THRESH_BINARY)
	i=0
	j=0

	while i < image.shape[0]-40:
		j = 0
		while j < image.shape[1]-41:
			green_fundus = g_channel[i:i+40,j:j+41]
			red_fundus = r_channel[i:i+40,j:j+41]
			ret,g = cv2.threshold(green_fundus,(np.amax(green_fundus)+np.mean(green_fundus))/2,255,cv2.THRESH_BINARY)
			ret,r = cv2.threshold(red_fundus,(np.amax(red_fundus)+np.mean(red_fundus))/2,255,cv2.THRESH_BINARY)
			imagexx[i:i+40,j:j+41] = cv2.bitwise_and(g,r)
			j = j +41
		i = i + 40
	return imagexx



	#x = cv2.bitwise_and(g_channel,r_channel)

	# if(g_channel == r_channel).nonzero():
	# 	image_x = g_channel
	# else:
	# 	image_x = 0
	# for i in range(image_x.shape[0]):
	# 	for j in range(image_x.shape[1]):
	# 		if(g_channel[i,j] == r_channel[i,j]):
	# 			image_x[i,j] = g_channel[i,j]
	# 		else:
	# 			image_x[i,j] = 0
	return x







if __name__ == "__main__":
    pathFolder = "/home/sherlock/Internship@iit/exudate-detection/diaretdb1/"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    destinationFolder = "/home/sherlock/Internship@iit/exudate-detection/diaretdb1-hello-color/"

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



    
