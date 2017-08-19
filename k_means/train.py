import numpy as np
from sklearn import preprocessing
import random
# import urllib
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"
# # download the file
# raw_data = urllib.urlopen(url)
# # load the CSV file as a numpy matrix
print("fcuuk")
dataset = np.loadtxt('data.txt', delimiter=",")
print(dataset.shape,"dataset_shape")
# separate the data from the target attributes
# X_train = dataset[0:900,2:7]
# X_train_normalized = preprocessing.normalize(X_train,norm='l1')
# y_train = dataset[0:900,0]

# X_test = dataset[900:1200,2:7]
# X_test_normalized = preprocessing.normalize(X_test,norm='l1')
# y_test = dataset[900:1200,0]


X=dataset[:,0:4]
y=dataset[:,4]

print(dataset[100000:100130,:])
from sklearn.model_selection import train_test_split

#y_bin  = [1 if iter >=1 else iter for iter in y]

j  = random.randint(0,100)

X_train,X_test,y_train, y_test=train_test_split(X, y, test_size=0.5, random_state=j)
print(j,"random_seed")
#print(X_train)
#print(X_train_normalized - X_train)
#print(X_test)
#print(y_test)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5),algorithm="SAMME",n_estimators=100)
#clf = SVC(kernel = 'poly')
#clf = KNeighborsClassifier(n_neighbors = 5)
#clf = AdaBoostClassifier()

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train) 

y_predicted = clf.predict(X_test)
print (y_predicted)

print("accuracy")
print(accuracy_score(y_test, y_predicted))

from sklearn.metrics import confusion_matrix
print("confusion matrix")
print (confusion_matrix(y_test,y_predicted))


print("SUPPORT VECTOR MACINES")

clf = SVC(kernel = 'linear')
clf.fit(X_train, y_train) 

y_predicted = clf.predict(X_test)
print (y_predicted)

print("accuracy")
print(accuracy_score(y_test, y_predicted))

from sklearn.metrics import confusion_matrix
print("confusion matrix")
print (confusion_matrix(y_test,y_predicted))



