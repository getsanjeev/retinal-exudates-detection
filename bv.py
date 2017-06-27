import cv2
import numpy as np
import os
import csv

def identify_OD(image,green_channel,add_index):	
	newfin = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=2)
	mask = np.ones(newfin.shape[:2], dtype="uint8") * 255		
	y1, ycontours, yhierarchy = cv2.findContours(newfin.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
	prev_contour = ycontours[0]
	for cnt in ycontours:				
		if cv2.contourArea(cnt) >= cv2.contourArea(prev_contour):
			print("dfdf")
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
	#cv2.imshow("bd",green_channel)

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


def extract_bv(image):
	dim = (800,615)
	fundus = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)	
	b,green_fundus,r = cv2.split(fundus)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	contrast_enhanced_green_fundus = clahe.apply(green_fundus)

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
	xmask = np.ones(fundus.shape[:2], dtype="uint8") * 255
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
	destinationFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11-bloodvessel/"
	if not os.path.exists(destinationFolder):
		os.mkdir(destinationFolder)
	for file_name in filesArray:
		print(pathFolder+'/'+file_name)
		file_name_no_extension = os.path.splitext(file_name)[0]
		fundus = cv2.imread(pathFolder+'/'+file_name)
		#fundus = cv2.imread("sss.tif")
		new_fundus = fundus.copy()
		bv_image = extract_bv(fundus)
		line = line_of_symmetry(bv_image)	
		b,green_fundus,r = cv2.split(fundus)
		sub_image = green_fundus[line-120:line+120,:]
		ret,fin = cv2.threshold(sub_image,(np.mean(sub_image) + np.amax(sub_image))/2,255,cv2.THRESH_BINARY)							
		new_fin = identify_OD(fin,green_fundus,line-120)
		#break							
	#cv2.waitKey()
		cv2.imwrite(destinationFolder+file_name_no_extension+"_bloodvessel.jpg",new_fin)			
