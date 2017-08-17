import numpy as np
import cv2
from numba import jit
import os
from matplotlib import pyplot as plt
import math

@jit
def standard_deviation_image(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_output = clahe.apply(image)
	result = clahe_output.copy()
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			sub_image = clahe_output[i:i+9,j:j+9]
			var = np.var(sub_image)
			result[i:i+9,j:j+9] = var
			j = j+9
		i = i+9
	return result

def get_SD_data(sd_image):	
	feature_1 = np.reshape(sd_image, (sd_image.size,1))
	print(feature_1.shape)
	return feature_1

def get_DistanceFromOD_data(image, centre):
	my_image = image.copy()
	x_cor = centre[0]
	y_cor = centre[1]
	feature_5 = np.reshape(image, (image.size,1))
	k = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			feature_5[k] = math.abs(x_cor-i) + math.abs(y_cor-j)
			k = k+1
	return feature_5

def get_HUE_data(hue_image):
	feature_2 = np.reshape(hue_image,(hue_image.size,1))
	print(feature_2.shape)
	return feature_2

def get_INTENSITY_data(intensity_image):
	feature_3 = np.reshape(intensity_image,(intensity_image.size,1))
	print(feature_3.shape)
	return feature_3

def get_EDGE_data(edge_candidates_image):
	feature_4 = np.reshape(edge_candidates_image,(edge_candidates_image.size,1))
	print(feature_4.shape)
	return feature_4

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

def identify_OD(image):
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
	return (cx,cy)

def identify_OD_bv_density(blood_vessel_image):
	los = line_of_symmetry(blood_vessel_image)
	sub_image = blood_vessel_image[los-100:los+100,:]
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
		i = i + 30	
	print(los,index)
	return (index,los)


def generate_csv(hue_image, intensity_image, SD_image, edge_image):
	with open('kmeans_featues.csv', 'w') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow(['hue','intensity','standard_deviation','edges'])
	while i < hue_image.shape[0]:
		j = 0
		while j < hue_image.shape[1]:
			filewriter.writerow([hue_image[i,j],intensity_image[i,j],SD_image[i,j],edge_image[i,j]])
			j = j + 1
		i = i + 1

@jit
def calculate_entropy(image):
	entropy = image.copy()
	sum = 0
	i = 0
	j = 0
	while i < entropy.shape[0]:
		j = 0
		while j < entropy.shape[1]:
			sub_image = entropy[i:i+10,j:j+10]
			histogram = cv2.calcHist([sub_image],[0],None,[256],[0,256])
			sum = 0
			for k in range(256):
				if histogram[k] != 0:					
					sum = sum + (histogram[k] * math.log(histogram[k]))
				k = k + 1
			entropy[i:i+10,j:j+10] = sum
			j = j+10
		i = i+10
	ret2,th2 = cv2.threshold(entropy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	newfin = cv2.erode(th2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
	return newfin

@jit
def edge_pixel_image(image,bv_image):
	edge_result = image.copy()
	edge_result = cv2.Canny(edge_result,30,100)
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			if edge_result[i,j] == 255 and bv_image[i,j] == 255:
				edge_result[i,j] = 0
			j = j+1
		i = i+1
	newfin = cv2.dilate(edge_result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
	return newfin

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
	dilated = cv2.erode(blood_vessels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=1)
	#dilated1 = cv2.dilate(blood_vessels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
	blood_vessels_1 = cv2.bitwise_not(dilated)
	return blood_vessels_1


if __name__ == "__main__":
	pathFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11/"
	filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
	DestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/Base11_exudates_kmeans/"
	
	if not os.path.exists(DestinationFolder):
		os.mkdir(DestinationFolder)
	# with open('hemmorages12.csv', 'w') as csvfile:
	# 	filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	# 	filewriter.writerow(['image_name', 'hue','intensity','standard_deviation','edges'])
	for file_name in filesArray:
		print(pathFolder+'/'+file_name)		
		file_name_no_extension = os.path.splitext(file_name)[0]
		fundus = cv2.imread(pathFolder+'/'+file_name)
		hsv_fundus = cv2.cvtColor(fundus,cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(hsv_fundus)
		gray_scale = cv2.cvtColor(fundus,cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		contrast_enhanced_fundus = clahe.apply(gray_scale)
		#entropy = calculate_entropy(contrast_enhanced_fundus)
		bv_image = extract_bv(gray_scale)						
		var_fundus = standard_deviation_image(gray_scale)
		edge_feature_output = edge_pixel_image(gray_scale,bv_image)
		#fin_edge = cv2.bitwise_and(edge_candidates,entropy)		
		(cx,cy) = identify_OD_bv_density(bv_image)				
		center = (cx,cy)
		newfin = cv2.dilate(edge_feature_output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
		edge_candidates = cv2.erode(newfin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
		cv2.circle(edge_candidates,(cx,cy), 100, (0,0,0), -10)

		feature1 = get_SD_data(var_fundus)/255
		feature2 = get_HUE_data(h)/255
		feature3 = get_INTENSITY_data(contrast_enhanced_fundus)/255
		#feature5 = get_DistanceFromOD_data(center)

		#print(feature5.shape)

		#feature4 = get_EDGE_data(edge_candidates)/255		

		#print(feature1[500:510,:],feature2[500:510,:],feature3[500:510,:],feature4[500:510,:])
		
		#data = np.concatenate((feature2),axis=1)	
		#data = np.float32(data)
		feature2 = np.float32(feature2)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
		ret,label,center=cv2.kmeans(feature2,5,None,criteria,50,cv2.KMEANS_RANDOM_CENTERS)
		print(center)

		green = [0,255,0]
		blue = [255,0,0]
		red = [0,0,255]
		white = [255,255,255]
		black = [0,0,0]

		color = [white,black,red,green,blue]
		color = np.array(color,np.uint8)
		label = np.reshape(label, gray_scale.shape)
		y = color[label]
		y = np.uint8(y)
		#cv2.imwrite("kmeans.jpg",y)
		print("-----------x-------DONE-------x----------")
		#cv2.waitKey()			
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_candidate_exudates.jpg",edge_candidates)		
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_result_exudates_kmeans.jpg",y)
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_sd_result.jpg",var_fundus)

# X = np.random.randint(25,50,(25,2))
# Y = np.random.randint(60,85,(25,2))
# Z = np.vstack((X,Y))

# # convert to np.float32
# Z = np.float32(Z)

# # define criteria and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# # Now separate the data, Note the flatten()
# A = Z[label.ravel()==0]
# B = Z[label.ravel()==1]

# # Plot the data
# plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1],c = 'r')
# plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()