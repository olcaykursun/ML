#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nearest mean classifier, our version from scratch vs scikit's
Created on Fri Feb  3 20:45:36 2023
Modified on Fri June  17 22:14:13 2023

@author: okursun

Useful links (mentioned in the lecture video):
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
https://en.wikipedia.org/wiki/Nearest_centroid_classifier
https://numpy.org/doc/stable/user/basics.broadcasting.html
https://numpy.org/doc/stable/reference/generated/numpy.append.html
"""



import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
iris.target += 1
X_train, X_test, y_train, y_test = train_test_split(iris.data, 
                                                    iris.target, 
                                                    train_size=0.6, 
                                                    #random_state=123, 
                                                    stratify=iris.target)

classes = np.unique(y_train)
num_classes = len(classes)

means = {}
for c in classes :
    means[c] = np.mean(X_train[y_train==c], axis = 0)



#%%
correct_preds = 0
for test_vector, true_label in zip(X_test, y_test):
    min_so_far = np.inf
    cls_so_far = 1
    for c in classes:        
        dx = test_vector - means[c]   #find the distance to centroid of class c
        dx2 = dx**2
        dist2 = dx2.sum()
        if dist2 < min_so_far:
            cls_so_far = c
            min_so_far = dist2
    correct_preds += cls_so_far == true_label #compare with true class
print(correct_preds / len(X_test))


#%%
dists = []
for c in classes:        
    dx = X_test - means[c]
    dists.append(np.sum(dx**2, axis = 1))

idx_closest_center = np.argmin(dists, axis = 0) + 1

distance_of_closest_center = np.min(dists, axis = 0)
#print(distance_of_closest_center)
#for i in range(len(X_test)):
#    print(dists[idx_closest_center[i]][i], distance_of_closest_center[i])

accuracy = np.mean(idx_closest_center == y_test)
print(accuracy)


#%%
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
