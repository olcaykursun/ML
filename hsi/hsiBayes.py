#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 08:41:12 2022

@author: okursun
"""

from scipy.io import loadmat

#https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
data = loadmat('Salinas_corrected.mat')['salinas_corrected']

gt = loadmat('Salinas_gt.mat')['salinas_gt']

num_rows = data.shape[0]
num_cols = data.shape[1]
num_bands = data.shape[2]

print(f'Image Size: {(num_rows, num_cols)}\nNumber of Bands: {num_bands}')
print(f'Double check the ground-truth shape: {gt.shape}')

gt_array = gt.reshape((num_rows*num_cols,))
#gt_array = gt.reshape(num_rows*num_cols)

class_names = sorted(list(set(gt_array)))
print('Classes in GT', class_names)

from matplotlib import pyplot as plt
plt.hist(gt_array)
#plt.hist(gt.ravel())

import pandas as pd
counts = pd.Series(gt_array).value_counts().sort_index()
print(counts)

from collections import Counter
counts2 = Counter(gt_array)
print([(i, counts2[i]) for i in sorted(counts2.keys())])

list_gt = list(gt_array)
print(*[(i, list_gt.count(i)) for i in sorted(class_names)], sep = '\n')

#%%
from sklearn.model_selection import train_test_split
data_array = data.reshape((num_rows*num_cols, -1))

labeled_pixels = (gt_array > 0)
class_names.remove(0)

data_array = data_array[labeled_pixels]
gt_array = gt_array[labeled_pixels]

X_train, X_test, y_train, y_test = train_test_split(data_array, gt_array, test_size=0.3, random_state=123, stratify=gt_array)

#%%
#SIMPLY RUN FROM SCIKIT
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('NB:', (y_test == y_pred).mean())



#IMPLEMENTATION DETAILS
#%%
#Training Phase, get the distribution stats for each class 
#More like, for each variable in each class
separated = {}
for i in range(len(X_train)):
    label = y_train[i]
    instance = X_train[i]
    if (label not in separated):
        separated[label] = []
    separated[label].append(instance)


import numpy as np
N, dims = X_train.shape
priors = {}
stats = {}
for clas in class_names:
    clas_instances = np.array(separated[clas])
    clas_N = len(clas_instances)
    priors[clas] = clas_N / N
    stats[clas] = []
    for feature in range(dims):
        feature_values = clas_instances[:,feature]
        avg = sum(feature_values)/clas_N
        variance = sum([pow(x-avg,2) for x in feature_values])/(clas_N-1)
        stats[clas].append((avg, variance))

#%%
#Testing Phase
import math
correct_classification = 0
for test_instance, true_class in zip(X_test, y_test):
    posteriors = {}
    for clas in class_names:
        likelihood = 1
        for feature in range(0, dims, 10):
            ave, variance = stats[clas][feature]
            x = test_instance[feature]
            exponent = math.exp(-(pow(x-ave,2)/(2*variance)))
            likelihood = likelihood * (1/(math.sqrt(2*math.pi*variance)))*exponent
        posterior = priors[clas]*likelihood
        posteriors[clas] = posterior
    predicted_class = max(posteriors, key=posteriors.get)
    if predicted_class == true_class:
        correct_classification = correct_classification + 1

print(correct_classification/len(X_test))        

#%%
#for details of the math: https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes

import math
correct_classification = 0
for test_instance, true_class in zip(X_test, y_test):
    posteriors = {}
    for clas in class_names:
        log_likelihood = 0
        for feature in range(dims):
            ave, variance = stats[clas][feature]
            x = test_instance[feature]
            exponent = -pow(x-ave,2)/(2*variance)
            log_likelihood = log_likelihood + math.log(1/(math.sqrt(2*math.pi*variance)))+exponent
        posterior = math.log(priors[clas])+log_likelihood
        posteriors[clas] = posterior
    predicted_class = max(posteriors, key=posteriors.get)
    if predicted_class == true_class:
        correct_classification = correct_classification + 1

print(correct_classification/len(X_test))        
