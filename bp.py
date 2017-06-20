import cv2
import numpy as np
import os
import csv


if __name__ == "__main__":
    pathFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11/"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    destinationFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11-exudates/"

    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)
    for file_name in filesArray:

    	print(pathFolder+'/'+file_name)
    	file_name_no_extension = os.path.splitext(file_name)[0]
    	fundus = cv2.imread(pathFolder+'/'+file_name)
    	b,green_fundus,r = cv2.split(fundus)
    	cv2.imwrite(destinationFolder+file_name_no_extension+"_exudates.jpg",green_fundus)	



    
