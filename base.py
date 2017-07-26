import cv2
import numpy as np
import os
import csv
import time
import math
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

def exudates_detection(image):
	ret,exudate_candidate = cv2.threshold(image,(np.mean(image) + np.amax(image))/2,255,cv2.THRESH_BINARY)	
	return exudate_candidate

def color_exudate_detction(green_fundus,red_fundus):
	ret,g_channel = cv2.threshold(green_fundus,(np.amax(green_fundus)+np.mean(green_fundus))/2,255,cv2.THRESH_BINARY)
	ret,r_channel = cv2.threshold(red_fundus,(np.amax(red_fundus)+np.mean(red_fundus))/2,255,cv2.THRESH_BINARY)
	if(g_channel == r_channel).nonzero():
		image_x = g_channel
	else:
		image_x = 0
	return image_x

def identify_OD_bv_density(sub_image,green_channel,add_index):
	gc = green_channel.copy()
	col_index = 0
	i = 0
	index = 0
	density = -1
	rr = 0	
	while i < sub_image.shape[1]:
		x1,y1 = sub_image[:,i:i+50].nonzero()
		count = x1.shape[0]		
		if(density < count):
			density = count
			index = i
		i = i +20
	cx = index + 25
	cy = add_index + 120
	cv2.circle(gc,(cx,cy), 80, (0,255,0), -5)
	return gc



def identify_OD(image,green_channel,add_index):
	newfin = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=2)
	mask = np.ones(newfin.shape[:2], dtype="uint8") * 255
	y1, ycontours, yhierarchy = cv2.findContours(newfin.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	prev_contour = ycontours[0]
	for cnt in ycontours:
		if cv2.contourArea(cnt) >= cv2.contourArea(prev_contour):
			prev_contour = cnt
			cv2.drawContours(mask, [cnt], -1, 0, -1)
	M = cv2.moments(prev_contour)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	print(cx,cy)
	print(cx+add_index,cy)
	print(green_channel.shape,"qweert")
	cv2.circle(image,(cx,cy), 20, (0,255,0), -5)
	cv2.circle(green_channel,(int(cx),cy+add_index), 80, (0,255,0), -10)
	return green_channel



def line_of_symmetry(image):
	image_v = image.copy()
	line = 0
	prev_diff = image_v.size
	for i in range(20,image_v.shape[0]-20):
		x1, y1 = image_v[0:i,:].nonzero()
		x2, y2 = image_v[i+1:image_v.shape[0],:].nonzero()
		diff = abs(x1.shape[0] - x2.shape[0])
		if diff < prev_diff:
			prev_diff = diff
			line = i
		i = i + 35
	return line

def maskWhiteCounter (mask_input):
    counter = 0
    for r in range(mask_input.shape[0]):
        for c in range(mask_input.shape[1]):
            if mask_input.item(r, c) == 255:
                counter+=1
    return counter

def calculateVarianceImage(image):
	var_image = image.copy()
	while(i < image.shape[0] and j < image.shape[1]):
		sub_image = image[i:i+10,j:j+10]
		mean_of_sub_image = np.mean(sub_image)
		#var_image[i,j] = (1/image.size())*math.pow((image[i,j] - ))

def extract_bv(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	contrast_enhanced_green_fundus = clahe.apply(image)
	# applying alternate sequential filtering (3 times closing opening)
	r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
	f5 = clahe.apply(f4)

	# removing very small contours through area parameter noise removal
	ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
	mask = np.ones(f5.shape[:2], dtype="uint8") * 255
	im2, contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) <= 200:
			cv2.drawContours(mask, [cnt], -1, 0, -1)
	im = cv2.bitwise_and(f5, f5, mask=mask)
	ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
	newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

	# removing blobs of microaneurysm & unwanted bigger chunks taking in consideration they are not straight lines like blood
	# vessels and also in an interval of area
	fundus_eroded = cv2.bitwise_not(newfin)
	xmask = np.ones(image.shape[:2], dtype="uint8") * 255
	x1, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in xcontours:
		shape = "unidentified"
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
		if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
			shape = "circle"
		else:
			shape = "veins"
		if(shape=="circle"):
			cv2.drawContours(xmask, [cnt], -1, 0, -1)

	finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)
	blood_vessels = cv2.bitwise_not(finimage)
	return finimage





if __name__ == "__main__":
	pathFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11/"
	filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
	bvDestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11-bloodvessel/"
	exDestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11-exudate/"
	colDestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11-color/"
	odDestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11-od/"
	if not os.path.exists(bvDestinationFolder):
		os.mkdir(bvDestinationFolder)
	if not os.path.exists(exDestinationFolder):
		os.mkdir(exDestinationFolder)
	if not os.path.exists(colDestinationFolder):
		os.mkdir(colDestinationFolder)
	if not os.path.exists(odDestinationFolder):
		os.mkdir(odDestinationFolder)

	for file_name in filesArray:
		print(pathFolder+'/'+file_name)
		file_name_no_extension = os.path.splitext(file_name)[0]
		fundus = cv2.imread(pathFolder+'/'+file_name)
		dim = (800,615)
		fundus1 = cv2.resize(fundus, dim, interpolation = cv2.INTER_AREA)
		centerY = fundus1.shape[1]
		centerX = fundus1.shape[0]
		radius = 350
		b,green_fundus,r = cv2.split(fundus1)
		image_bin = color_exudate_detction(green_fundus,r)
		#image_bin1 = crop_circle(image_bin,radius,centerX,centerY)
		cv2.imwrite(colDestinationFolder+file_name_no_extension+"_color.jpg",image_bin)
		new_fundus = green_fundus.copy()
		bv_image = extract_bv(new_fundus)
		cv2.imwrite(bvDestinationFolder+file_name_no_extension+"_bloodvessel.jpg",bv_image)
		line = line_of_symmetry(bv_image)
		sub_image = green_fundus[line-120:line+120,:]
		bv_sub_image = bv_image[line-120:line+120,:]
		ret,fin = cv2.threshold(sub_image,(np.mean(sub_image) + np.amax(sub_image))/2,255,cv2.THRESH_BINARY)
		od_image_through_bv = identify_OD_bv_density(bv_sub_image,green_fundus,line-120)
		new_fin = identify_OD(fin,od_image_through_bv,line-120)
		cv2.imwrite(odDestinationFolder+file_name_no_extension+"_od.jpg",new_fin)
		exu_cand = exudates_detection(new_fin)
		#qimage = crop_circle(exu_cand,radius,centerX,centerY)
		cv2.imwrite(exDestinationFolder+file_name_no_extension+"_exudates.jpg",exu_cand)				
