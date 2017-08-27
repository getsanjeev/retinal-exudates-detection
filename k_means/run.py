import numpy as np
import cv2
from numba import jit
import os
from matplotlib import pyplot as plt
import math
import csv
from sklearn import preprocessing
import numpy as np
from sklearn import preprocessing
import random
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

print(name_array,"HEY NAMES")	
os.rename("test.csv", "test.txt")	
dataset_train = np.loadtxt('train.txt', delimiter=",")
print(dataset_train.shape,"train_shape")	
dataset_test = np.loadtxt('test.txt', delimiter=",")
print(dataset_test.shape,"test_shape")
X_train = dataset_train[:,0:5]
Y_train = dataset_train[:,5]
X_test = dataset_test[:,0:5]
Y_test = dataset_test[:,5]
print(dataset_train[100000:100130,:])


#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5),algorithm="SAMME",n_estimators=100)
#clf = SVC(kernel = 'poly')
#clf = KNeighborsClassifier(n_neighbors = 5)
#clf = AdaBoostClassifier()

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, Y_train) 
Y_predicted = clf.predict(X_test)
print (Y_predicted.shape)
print("accuracy")
print(accuracy_score(Y_test, Y_predicted))
print("confusion matrix")
print (confusion_matrix(Y_test,Y_predicted))
print("renamed files and all....");
print(name_array)
print(len(name_array))
#	show_final_result(name_array)
i = 0
j = 0
lc = 0
counter = 0
k = 0
while k < len(name_array):		
	edge_candidates = cv2.imread(DestinationFolder+name_array[k]+"_edge_candidates.jpg")
	print("printing",DestinationFolder+name_array[k]+"_edge_candidates.jpg")
	result = edge_candidates.copy()
	while i < edge_candidates.shape[0]:
		j = 0
		while j < edge_candidates.shape[1]:
			if int(edge_label[lc,0])==1:
				result[i,j] = Y_predicted[counter]
				counter = counter + 1
			lc = lc +1
			j = j + 1
		i = i + 1
	k = k +1
	print("value of counter",counter)
	cv2.imwrite(DestinationFolder+name_array[k]+"final_result.jpg",result)
	print("fcuk",k)
