import numpy as np
import cv2
from numba import jit
import os
from matplotlib import pyplot as plt
import math
import csv
from sklearn import preprocessing
from numba import jit


@jit
def calC_accuracy(result, label):
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	i = 0
	j = 0

	print(np.unique(result))
	print(np.unique(label))

	while i < result.shape[0]:
		j = 0
		while j < result.shape[1]:
			if label[i,j] == 255:
				if result[i,j] == label[i,j]:
					tp = tp + 1
				else:
					fn = fn + 1
			else:
				if result[i,j] == label[i,j]:
					tn = tn + 1
				else:
					fp = fp + 1
			j = j + 1
		i = i + 1
	print("TN =",tn,"FP =",fp)
	print("FN =",fn,"TP =",tp)
	print("Sensitivity = ",float(tp/(tp+fn+1)))
	print("Specificity = ",float(tn/(tn+fp+1)))
	print("Accuracy = ",float((tn+tp)/(fn+fp+1+tn+tp)))
	print("PPV = ",float(tp/(tp+fp+1)))
	return float(tp/(tp+fp+1))



if __name__ == "__main__":
	pathFolder = "/home/sherlock/Internship@iit/exudate-detection/knn_results-exudates/"
	filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
	LabelFolder = "/home/sherlock/Internship@iit/exudate-detection/diaretdb1-label/"		
	print("WELCOME")
	array = {}
	sumd = 0
		
	for file_name in filesArray:		
		file_name_no_extension = os.path.splitext(file_name)[0]		
		print(file_name_no_extension[0:8])						
		fundus1 = cv2.imread(pathFolder+'/'+file_name_no_extension[0:8]+"knn_result.bmp")
		print(pathFolder+'/'+file_name_no_extension[0:8]+"ab_result.bmp")
		label_image = cv2.imread(LabelFolder+'/'+file_name_no_extension[0:8]+"_final_label.bmp")
		print(LabelFolder+'/'+file_name_no_extension[0:8]+"_final_label.bmp")
		ppv = calC_accuracy(fundus1[:,:,0],label_image[:,:,0])
		sumd = sumd + ppv		
		array[file_name_no_extension[5:8]] = ppv
	print(sumd/36,"PPV average")
	centers = range(len(array))
	plt.bar(centers, array.values(), align='center', tick_label=array.keys())
	plt.xlim([0, 36])
	plt.ylabel('PPV value')
	plt.xlabel('Image no in DIARETDB1')
	plt.show()




		