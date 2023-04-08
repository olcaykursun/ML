#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 20:51:56 2023

@author: okursun
"""

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

import matplotlib.pyplot as plt

plt.figure()
plt.hist(X_train[:,1], density=False, bins=2)

plt.figure()
plt.hist(X_train[:,1], density=False, bins=10)

# plt.figure()
# plt.hist(X_train[:,1], density=False, bins=100)

n_bins = 10
from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
X_train = est.fit_transform(X_train)
X_test = est.transform(X_test)

clf = CategoricalNB(alpha=1)
clf.fit(X_train, y_train)
print('acc =', clf.score(X_test, y_test))

clf = MultinomialNB(alpha=1)
clf.fit(X_train, y_train)
print('acc =', clf.score(X_test, y_test))

clf = GaussianNB()
clf.fit(X_train, y_train)
print('acc =', clf.score(X_test, y_test))

#%%
alpha = 1

N, n_dims = X_train.shape
num_classes = len(iris.target_names)
priors = np.zeros(num_classes)
stats = np.zeros((num_classes, n_dims, n_bins))  
for clas in range(num_classes):
    clas_instances = X_train[y_train==clas]
    clas_N = len(clas_instances)
    priors[clas] = clas_N / N
    for d in range(n_dims):
        for b in range(n_bins):
            stats[clas, d, b] = (clas_instances[:,d]==b).sum() + alpha
    stats[clas] = stats[clas] / (clas_N + n_bins*alpha)

#%%
correct_classification = 0
for test_instance, true_class in zip(X_test, y_test):
    posteriors = np.zeros(num_classes)
    for clas in range(num_classes):
        likelihood = 1
        for feature in range(n_dims):
            x = test_instance[feature]
            likelihood = likelihood * stats[clas][feature][int(x)]
        posteriors[clas] = priors[clas]*likelihood
    predicted_class = np.argmax(posteriors)
    if predicted_class == true_class:
        correct_classification = correct_classification + 1
print(correct_classification/len(y_test))
