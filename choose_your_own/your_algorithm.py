#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from datetime import datetime

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

print len(features_train[0])
print len(features_train)
print len(features_test[0])
print len(features_test)


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
#plt.close()
################################################################################


#quit()
### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


t0 = datetime.now()
print t0
clf =SVC(kernel = 'rbf', C = 100000)
# clf = AdaBoostClassifier(n_estimators=75)

print datetime.now()
clf.fit(features_train,labels_train)
t1 = datetime.now()
print "time to fit:",(t1-t0)
pred = clf.predict(features_test)
error = accuracy_score(labels_test,pred)
t2 = datetime.now()
print "time to predict",(t2-t1)

print "error:", error

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
