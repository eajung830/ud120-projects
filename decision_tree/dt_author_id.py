#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
sys.path.append("../choose_your_own/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print len(features_train[2]),len(features_train)

#quit()

#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#for val in [10,100,1000,10000]:


clf = DecisionTreeClassifier(min_samples_split=40)
t0 = time()
print t0
clf.fit(features_train, labels_train)
print "time to fit:", round(time()-t0,3), "seconds"
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print "accuracy=",accuracy

print pred[10],pred[26],pred[50]
print pred.tolist().count(1)
print len(pred.tolist())

#########################################################


