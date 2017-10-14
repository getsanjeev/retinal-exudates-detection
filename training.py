import numpy as np
import cv2
from numba import jit
import os
from matplotlib import pyplot as plt
import math
import csv
from sklearn import preprocessing



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
			sub_image = clahe_output[i:i+20,j:j+25]
			var = np.var(sub_image)
			result[i:i+20,j:j+25] = var
			j = j+25
		i = i+20
	return result

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

def get_DistanceFromOD_data(image, centre):
	my_image = image.copy()
	x_cor = centre[0]
	y_cor = centre[1]
	feature_5 = np.reshape(image, (image.size,1))
	k = 0
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			feature_5[k] = math.fabs(x_cor-i) + math.fabs(y_cor-j)
			j = j+1
			k = k+1
		i = i+1
	return feature_5

@jit
def remove_bv_image(image,bv_image):
	edge_result = image[:,:,0]				
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			if edge_result[i,j] == 255 and bv_image[i,j] == 255:
				edge_result[i,j] = 0
			j = j+1
		i = i+1
	return edge_result


def count_ones(image,value):
	i = 0
	j = 0 
	k = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			if int(image[i,j]) == value:
				k = k+1
			j = j + 1			
		i = i+1
	return k


def get_average_intensity(green_channel):
	average_intensity = green_channel.copy()
	i = 0
	j = 0
	while i < green_channel.shape[0]:
		j = 0
		while j < green_channel.shape[1]:
			sub_image = green_channel[i:i+20,j:j+25]
			mean = np.mean(sub_image)
			average_intensity[i:i+20,j:j+25] = mean
			j = j+25
		i = i+20
	result = np.reshape(average_intensity, (average_intensity.size,1))
	return result

def get_average_hue(hue_image):
	average_hue = hue_image.copy()
	i = 0
	j = 0
	while i < hue_image.shape[0]:
		j = 0
		while j < hue_image.shape[1]:
			sub_image = hue_image[i:i+20,j:j+25]
			mean = np.mean(sub_image)
			average_hue[i:i+20,j:j+25] = mean
			j = j+25
		i = i+20
	result = np.reshape(average_hue, (average_hue.size,1))
	return result

def get_average_saturation(hue_image):
	average_hue = hue_image.copy()
	i = 0
	j = 0
	while i < hue_image.shape[0]:
		j = 0
		while j < hue_image.shape[1]:
			sub_image = hue_image[i:i+20,j:j+25]
			mean = np.mean(sub_image)
			average_hue[i:i+20,j:j+25] = mean
			j = j+25
		i = i+20
	result = np.reshape(average_hue, (average_hue.size,1))
	return result



def get_SD_data(sd_image):	
	feature_1 = np.reshape(sd_image, (sd_image.size,1))
	return feature_1

def get_HUE_data(hue_image):	
	feature_2 = np.reshape(hue_image,(hue_image.size,1))	
	return feature_2

def get_saturation_data(s_image):
	feature = np.reshape(s_image,(s.size,1))	
	return feature


def get_INTENSITY_data(intensity_image):	
	feature_3 = np.reshape(intensity_image,(intensity_image.size,1))	
	return feature_3

def get_EDGE_data(edge_candidates_image):
	feature_4 = np.reshape(edge_candidates_image,(edge_candidates_image.size,1))	
	return feature_4

def get_RED_data(red_channel):	
	feature_5 = np.reshape(red_channel, (red_channel.size,1))	
	return feature_5

def get_GREEN_data(green_channel):
	feature_6 = np.reshape(green_channel, (green_channel.size,1))	
	return feature_6


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
	pathFolder = "/home/sherlock/Internship@iit/exudate-detection/training/"
	filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
	DestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/training-results/"
	LabelFolder = "/home/sherlock/Internship@iit/exudate-detection/diaretdb1-label/"	
	
	if not os.path.exists(DestinationFolder):
		os.mkdir(DestinationFolder)	

	qq = 0

	OD_data = np.genfromtxt('OD_info.txt', delimiter=',', dtype=None, names=('name','x-coordinate','y-coordinate'))
	coordinates = []
	name = []	
	counterd = 0

	for t in OD_data:		
		coordinates.append((t[1],t[2]))
		name.append(t[0].decode("utf-8"))
		counterd = counterd + 1
		
	for file_name in filesArray:		
		file_name_no_extension = os.path.splitext(file_name)[0]
		coordinates_OD = coordinates[name.index(file_name_no_extension+"_resized")]		
		print(pathFolder+'/'+file_name,"Read this image",file_name_no_extension)
		fundus1 = cv2.imread(pathFolder+'/'+file_name)
		fundus = cv2.resize(fundus1,(800,615))		
		fundus_mask = cv2.imread("fmask.tif")
		fundus_mask = cv2.resize(fundus_mask,(800,615))
		f1 = cv2.bitwise_and(fundus[:,:,0],fundus_mask[:,:,0])
		f2 = cv2.bitwise_and(fundus[:,:,1],fundus_mask[:,:,1])
		f3 = cv2.bitwise_and(fundus[:,:,2],fundus_mask[:,:,2])
		fundus_dash = cv2.merge((f1,f2,f3))

		cv2.imwrite(DestinationFolder+file_name_no_extension+"_resized_fundus.bmp",fundus_dash)	

		b,g,r = cv2.split(fundus_dash)		
		hsv_fundus = cv2.cvtColor(fundus_dash,cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(hsv_fundus)		
		gray_scale = cv2.cvtColor(fundus_dash,cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		contrast_enhanced_fundus = clahe.apply(gray_scale)		
		contrast_enhanced_green_fundus = clahe.apply(g)

		average_intensity = get_average_intensity(contrast_enhanced_green_fundus)/255
		average_hue = get_average_hue(h)/255
		average_saturation = get_average_saturation(s)/255	
		
		bv_image_dash = extract_bv(g)
		bv_image = extract_bv(gray_scale)

		cv2.imwrite(DestinationFolder+file_name_no_extension+"_blood_vessels.bmp",bv_image_dash)
		var_fundus = standard_deviation_image(contrast_enhanced_fundus)
		edge_feature_output = edge_pixel_image(gray_scale,bv_image)		
		newfin = cv2.dilate(edge_feature_output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
		edge_candidates = cv2.erode(newfin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)		
		edge_candidates = np.uint8(edge_candidates)	
															
		label_image = cv2.imread(LabelFolder+'/'+file_name_no_extension+"_final_label.bmp")

		deviation_matrix = deviation_from_mean(gray_scale)

		feature1 = get_SD_data(var_fundus)/255
		feature2 = get_HUE_data(h)/255
		feature3 = get_saturation_data(s)/255
		feature4 = get_INTENSITY_data(contrast_enhanced_fundus)/255
		feature5 = get_RED_data(r)/255
		feature6 = get_GREEN_data(g)/255
		feature7 = get_DistanceFromOD_data(bv_image,coordinates_OD)/(var_fundus.shape[0]+var_fundus.shape[1])
		feature8 = get_HUE_data(deviation_matrix)/255
		print(feature8.shape,"deviation data shape")


		Z = np.hstack((feature2,feature3))	#HUE and SATURATION
		Z = np.float32(Z)

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
		ret,label,center=cv2.kmeans(Z,6,None,criteria,50,cv2.KMEANS_RANDOM_CENTERS)	

		u, indices, counts = np.unique(label, return_index=True, return_counts=True)	
		
		center_t = [(t[0]*255,t[1]*255) for t in center]		
		ex_color = (40,230)

		distance = [(abs(t[0]- ex_color[0]),t) for t in center_t]								
		index1 = distance.index((min(distance)))
		if counts[distance.index((min(distance)))] > 0.2*gray_scale.shape[0]*gray_scale.shape[1]:
			index1 = -1		

		distance2 = [(abs(t[0]- ex_color[0])+abs(t[1]-ex_color[1]),t) for t in center_t]		
		index2 = -1		
		if min(distance2)[0] <=25:
			index2 = distance2.index((min(distance2)))		
		if counts[distance2.index((min(distance2)))] > 0.2*gray_scale.shape[0]*gray_scale.shape[1]:
			index2 = -1

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
		if index1 == -1:
			test.fill(0)
		else:
			test[test!=distance.index((min(distance)))] = -1
			test[test==distance.index((min(distance)))] = 255
			test[test==-1] = 0

		test2 = label.copy()
		if index2 == -1:
			test2.fill(0)
		else:
			test2[test2!=index2] = -1
			test2[test2==index2] = 255
			test2[test2==-1] = 0
	

		y = color[label]
		y = np.uint8(y)

		res_from_clustering = np.bitwise_or(test2,test)
		
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_candidate_exudates.bmp",edge_candidates)	
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_result_exudates_kmeans.bmp",y)	
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_test_result.bmp",test)
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_test2_result.bmp",test2)		
		final_candidates = np.bitwise_or(edge_candidates,res_from_clustering)	

		OD_loc = gray_scale.copy()
		cv2.circle(OD_loc,coordinates_OD, 70, (0,0,0), -10)
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_OD_.bmp",OD_loc)

		cl_res_dev = cv2.imread("/home/sherlock/Internship@iit/exudate-detection/training-result-kmeans-deviation/"+file_name_no_extension+"_final_candidates.bmp")
		cl_res_dev = remove_bv_image(cl_res_dev,bv_image_dash)
		cv2.imwrite(DestinationFolder+file_name_no_extension+"removed_bv_from.bmp",cl_res_dev)
		print("/home/sherlock/Internship@iit/exudate-detection/training-result-kmeans-deviation/"+file_name_no_extension+"_final_candidates.bmp")		
		final_candidates = np.bitwise_or(final_candidates,cl_res_dev)
				
		cv2.circle(final_candidates,coordinates_OD, 70, (0,0,0), -10)
		maskk = cv2.imread("MASK.bmp")
		final_candidates = np.bitwise_and(final_candidates,maskk[:,:,0])
		
		final_candidates = final_candidates.astype('uint8')
		final_candidates = cv2.dilate(final_candidates, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)		
		cv2.imwrite(DestinationFolder+file_name_no_extension+"_final_candidates.bmp",final_candidates)
		
		candidates_vector = np.reshape(final_candidates,(final_candidates.size,1))/255
		#print(final_candidates.shape,"SHAPE OF FINAL CANDIDATE")		
		
		b,gg,r = cv2.split(label_image)
		label = np.reshape(gg,(gg.size,1))/255
		co3 = count_ones(edge_candidates,255)
		no_of_white = count_ones(label,1)
		#print(no_of_white,"no of white pixels")
		#print(co3,"check me")
		temp = 0
		counter = 0
		this_image_rows = 0
		with open('training.csv', 'a') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			while counter < feature1.shape[0]:
				if candidates_vector[counter,0] == 1:
					qq = qq + 1
					temp = counter
					this_image_rows = this_image_rows+1
					filewriter.writerow([feature2[counter,0],feature3[counter,0],feature4[counter,0],feature5[counter,0],feature6[counter,0],feature7[counter,0],feature8[counter,0],average_intensity[counter,0],average_hue[counter,0],average_saturation[counter,0],int(label[counter,0])])
				counter = counter + 1
						
		print("no of rows addded : ", this_image_rows)

	print("no of pixxels in total", qq)
	os.rename("training.csv","training.txt")