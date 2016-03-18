#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#for val in [10,100,1000,10000]:
for val in [10000]:
    clf = SVC(kernel = 'rbf', C=val)
    t0 = time()
    print t0
    clf.fit(features_train, labels_train)
    print "time to fit:", round(time()-t0,3), "seconds"
    pred = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    print "C =",val,"accuracy=",accuracy

print pred[10],pred[26],pred[50]
print pred.tolist().count(1)
print len(pred.tolist())

#########################################################


