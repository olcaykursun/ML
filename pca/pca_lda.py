#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:24:20 2023

@author: okursun
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
print(digits.data.shape)

X_train, X_test, y_train, y_test = train_test_split(digits.data, 
                                                    digits.target, 
                                                    train_size=0.1)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print('KNN accuracy = ', neigh.score(X_test, y_test))

pca = PCA(n_components=25)
X_train_reduced_by_PCA = pca.fit_transform(X_train)
X_test_reduced_by_PCA = pca.transform(X_test)

n_components = 5

neigh.fit(X_train_reduced_by_PCA[:,:n_components], y_train)
print('PCA-KNN accuracy = ', neigh.score(X_test_reduced_by_PCA[:,:n_components], y_test))

lda = LinearDiscriminantAnalysis(n_components=n_components)
lda.fit(X_train, y_train)
print('LDA accuracy = ', lda.score(X_test, y_test))

neigh.fit(lda.transform(X_train), y_train)
print('LDA-KNN accuracy = ', neigh.score(lda.transform(X_test), y_test))

lda.fit(X_train_reduced_by_PCA, y_train)
print('PCA-LDA accuracy = ', lda.score(X_test_reduced_by_PCA, y_test))
neigh.fit(lda.transform(X_train_reduced_by_PCA), y_train)
print('PCA-LDA-KNN accuracy = ', neigh.score(lda.transform(X_test_reduced_by_PCA), y_test))
