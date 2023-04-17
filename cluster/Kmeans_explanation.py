#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 21:47:32 2023

@author: okursun
"""

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  
y = iris.target

#First reduce dimensionality from 4 to 2 for better display
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k', s=40) #use class-labels for colors
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")

n_clusters = 3
clustering = KMeans(n_clusters=n_clusters, init = 'random', n_init = 10)
clustering.fit(X_reduced)
clusters = clustering.labels_
plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, edgecolor='k', s=40) #use cluster indices for colors
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")

#%%
print('ARI', adjusted_rand_score(y, clusters))
