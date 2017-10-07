import cv2
import os
import csv
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append((x, y))

if __name__ == "__main__":
	pathFolder = "/home/sherlock/Internship@iit/exudate-detection/diaretdb_resized/"
	filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]				
	name = []
	counter = 0
			
	for file_name in filesArray:		
		file_name_no_extension = os.path.splitext(file_name)[0]
		name.append(file_name_no_extension)
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", click_and_crop)
		image = cv2.imread(pathFolder+'/'+file_name)
		print(pathFolder+'/'+file_name)
		cv2.imshow("image",image)		
		cv2.waitKey()

		with open('OD_info.csv', 'a') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			filewriter.writerow([name[counter],refPt[counter]])
			counter = counter +1
	print(counter,"no of files")
	os.rename("OD_info.csv","OD_info.txt")
	data_OD = np.loadtxt('OD_info.txt', delimiter=",")
	print(data_OD)

		#print(refPt)
	


 
	# # check to see if the left mouse button was released
	# elif event == cv2.EVENT_LBUTTONUP:
	# 	# record the ending (x, y) coordinates and indicate that
	# 	# the cropping operation is finished
	# 	refPt.append((x, y))
	# 	cropping = False
 
	# 	# draw a rectangle around the region of interest
	# 	cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
	# 	cv2.imshow("image", image)

 
# load the image, clone it, and setup the mouse callback function
# image = cv2.imread("download.png")
