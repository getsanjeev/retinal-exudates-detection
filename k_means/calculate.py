import numpy as np
from sklearn import preprocessing
import random
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy
import cv2
import os


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


os.rename("test.csv","test.txt")
os.rename("train.csv","train.txt")
print("fcuuk")
dataset_train = np.loadtxt('train.txt', delimiter=",")
#dataset_train = pd.read_csv("train.csv")
print(dataset_train.shape,"train_shape")

dataset_test = np.loadtxt('test.txt', delimiter=",")
#dataset_test = pd.read_csv("test.csv")
print(dataset_test.shape,"test_shape")

X_train = dataset_train[:,0:7]
Y_train = dataset_train[:,8]

print(dataset_train[50:55,8])

X_test = dataset_test[:,0:7]
Y_test = dataset_test[:,8]



print(dataset_train[100000:100130,:])


#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5),algorithm="SAMME",n_estimators=100)
#clf = SVC(kernel = 'poly')
#clf = KNeighborsClassifier(n_neighbors = 5)
#clf = AdaBoostClassifier()

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, Y_train) 

Y_predicted = clf.predict(X_test)
print (Y_predicted)

print("accuracy")
print(accuracy_score(Y_test, Y_predicted))

print("confusion matrix")
print (confusion_matrix(Y_test,Y_predicted))

print(Y_predicted.shape,"shape of y predicted")


DestinationFolder = "/home/sherlock/Internship@iit/exudate-detection/testing-results/"
name_array = ['image069', 'image056', 'image043', 'image063', 'image038', 'image050', 'image039', 'image015', 'image085', 'image067', 'image086', 'image003', 'image013', 'image037', 'image044', 'image025', 'image080', 'image068', 'image009', 'image002', 'image079', 'image008', 'image062', 'image074', 'image057', 'image081', 'image033', 'image014', 'image032', 'image051', 'image075', 'image001', 'image020', 'image007', 'image031', 'image049', 'image087', 'image027', 'image019', 'image045', 'image026', 'image055', 'image061', 'image021', 'image073']
resultFolder = "/home/sherlock/Internship@iit/exudate-detection/results-exudates/"	
print(len(name_array))
if not os.path.exists(resultFolder):
	os.mkdir(resultFolder)
size_m = 0
i = 0
j = 0
lc = 0
while size_m < len(name_array):
	current = cv2.imread(DestinationFolder+name_array[size_m]+"_edge_candidates.bmp")
	print(DestinationFolder+name_array[size_m]+"_edge_candidates.bmp")
	cv2.imshow("iodjbj",current)
	print("current ka size",current.shape)
	x,current_m,z = cv2.split(current)
	print(count_ones(current_m,255),"now again check",name_array[size_m])
	i = 0
	while i < current_m.shape[0]:
		j = 0
		while j < current_m.shape[1]:
			if current_m[i,j] == 255:
				current_m[i,j] = 255*Y_predicted[lc]
				lc = lc + 1
			j = j + 1
		i = i + 1
	cv2.imwrite(resultFolder+name_array[size_m]+"_result.bmp",current_m)
	size_m = size_m + 1

print("DONE_-------------------x----xxxxx-xx-x")








