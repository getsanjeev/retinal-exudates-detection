import cv2
import numpy as np
import os
import csv



def remove_OD(image):
	image_x = image.copy()
	OP1 = cv2.morphologyEx(image_x, cv2.MORPH_CLOSE, 
		cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations = 1)
	ret3,otsu_result = cv2.threshold(OP1,(np.amax(OP1)+np.mean(OP1))/2 + 20,255,cv2.THRESH_BINARY_INV)
	print(ret3)





	return otsu_result




def add_salt_and_pepper_noise(image):
	s_p_ratio = 0.5
	quantity = 0.005
	noisy_image = image.copy()
	num_salt_points = np.ceil(image.size*quantity*s_p_ratio)
	random_salt_coordinates_x = np.random.randint(image.shape[0],size = int(num_salt_points))
	random_salt_coordinates_y = np.random.randint(image.shape[1],size = int(num_salt_points))
	for i, j in zip(random_salt_coordinates_x, random_salt_coordinates_y):
		noisy_image[i,j] = 0

	num_pepper_points = np.ceil(image.size*quantity*(1-s_p_ratio))
	random_pepper_coordinates_x = np.random.randint(image.shape[0],size = int(num_pepper_points))
	random_pepper_coordinates_y = np.random.randint(image.shape[1],size = int(num_pepper_points))
	for i, j in zip(random_pepper_coordinates_x, random_pepper_coordinates_y):
		noisy_image[i,j] = 255
	return noisy_image






if __name__ == "__main__":
    pathFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11/"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    destinationFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11-exudates/"

    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)
    for file_name in filesArray:
    	print(pathFolder+'/'+file_name)    
    	fundus = cv2.imread(pathFolder+'/'+file_name)
    	file_name_no_extension = os.path.splitext(file_name)[0]
    	hsi = cv2.cvtColor(fundus, cv2.COLOR_BGR2HSV)
    	h,s,v = cv2.split(hsi)
    	noisy_image = add_salt_and_pepper_noise(v)
    	filtered_image = cv2.medianBlur(noisy_image,3)
    	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    	clahe_image = clahe.apply(filtered_image)
    	shade_corrected_image = remove_OD(clahe_image)
    	cv2.imwrite(destinationFolder+file_name_no_extension+"_exudates.jpg",shade_corrected_image)



    
