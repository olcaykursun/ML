#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:54:15 2023

@author: okursun
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

X_train, X_test, y_train, y_test = train_test_split(digits.data, 
                                                    digits.target, 
                                                    train_size=0.6)

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=10)       
X_train_reduced_by_PCA = pca.fit_transform(X_train)
X_test_reduced_by_PCA = pca.transform(X_test)

recons_train = pca.inverse_transform(X_train_reduced_by_PCA)
recons_test = pca.inverse_transform(X_test_reduced_by_PCA)

residual_train = X_train-recons_train
residual_test = X_test-recons_test

vmin, vmax = X_train.min(), X_train.max()
vmin1, vmax1 = recons_train.min(), recons_test.max()

#%%

plt.figure();
plt.imshow(X_train[10].reshape(8, 8), cmap=plt.cm.gray, vmin=vmin, vmax=vmax);
plt.figure();
plt.imshow(recons_train[10].reshape(8, 8), cmap=plt.cm.gray, vmin=vmin, vmax=vmax);
plt.figure()
plt.imshow(residual_train[10,:].reshape(8, 8), cmap=plt.cm.gray, vmin=-vmax, vmax=vmax);

#%%
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(100,50))
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

reduced_clf = MLPClassifier(hidden_layer_sizes=(100,50))
reduced_clf.fit(X_train_reduced_by_PCA, y_train)
print(reduced_clf.score(X_test_reduced_by_PCA, y_test))

#%%
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

reduced_clf = DecisionTreeClassifier()
reduced_clf.fit(X_train_reduced_by_PCA, y_train)
print(reduced_clf.score(X_test_reduced_by_PCA, y_test))
