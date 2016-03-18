#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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

#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# create predictor object
clf = GaussianNB()
# fit object to training
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0,3),"s"
# get predictions for test set
t0 = time()
labels_test_pred = clf.predict(features_test)
print "prediction time:", round(time()-t0,3),"s"

# get accuracy
accuracy = accuracy_score(labels_test, labels_test_pred)

print(accuracy)

#########################################################
'''
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    ### create classifier
    clf = GaussianNB() #TODO

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    #TODO

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test) #TODO


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = accuracy_score(l
'''
