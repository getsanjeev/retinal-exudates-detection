import cv2
import numpy as np
import math
import os

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
			sub_image = clahe_output[i:i+3,j:j+3]
			mean = np.mean(sub_image)
			sub_image = sub_image - mean
			result[i:i+3,j:j+3] = sub_image
			j = j+3
		i = i+3
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
	print(np.reshape(feature_5,(10,10)))
	return feature_5


#os.remove("model.sh")

fundus_mask = cv2.imread("fmask.tif")
fundus_mask = cv2.resize(fundus_mask,(800,615))
image = fundus_mask[:,:,0]

row = image.shape[0]
col = image.shape[1]
print(row,col)
image.fill(0)
i = 0
j = 0
count = 0
while i < row:
	j = 0
	while j < col:
		if math.sqrt(math.pow(abs(i -307),2)+ math.pow(abs(j-400),2)) <= 290:
			image[i,j] = 255
			count=  count + 1
		j = j+1
	i = i+1
print(count)
cv2.imwrite("MASK.bmp",image)

# im = cv2.imread("download.png")
# gray_scale = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# print(get_DistanceFromOD_data(gray_scale,(2,3)))







# xx = deviation_from_mean(gray_scale)
# print("deviation matrix")
# print(xx)
# xx = np.reshape(xx,(xx.size,1))
# print(xx.shape)

# Z = np.hstack((xx))
# Z = np.float32(Z)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
# ret,label,center=cv2.kmeans(Z,6,None,criteria,50,cv2.KMEANS_RANDOM_CENTERS)
# print("center of clusters : ")
# print(center)

# u, indices, counts = np.unique(label, return_index=True, return_counts=True)
# print(u,"unique labels in Clustered image")
# print(counts,"count of labels in Clustered image")
		
# center_t = [(t[0]) for t in center]
# print(center_t,"centre relevant to me")
# print(max(center_t),"max of center")

# green = [0,255,0]
# blue = [255,0,0]
# red = [0,0,255]
# white = [255,255,255]
# black = [0,0,0]
# pink = [220,30,210]
# sky = [30,240,230]
# yellow = [230,230,30]
# color = [white,black,red,green,blue,pink]
# color = np.array(color,np.uint8)
# label = np.reshape(label, gray_scale.shape)


# test = label.copy()				
# test[test!=u[center_t.index(max(center_t))]] = -1
# test[test==u[center_t.index(max(center_t))]] = 255
# test[test==-1] = 0

# y = color[label]
# y = np.uint8(y)

# print("color se labelled")
# #print(y)

# cv2.imwrite("_result_exudates_kmeans.bmp",y)	
# 		#cv2.imwrite(DestinationFolder+file_name_no_extension+"_test_result.bmp",test)				
# cv2.imwrite("_final_candidates.bmp",test)