#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 11:29:25 2023

@author: okursun
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

#7 classes, 325 dimensions and 73 examples
#https://home.penglab.com/proj/mRMR/
filename = 'test_lung_s3.csv'
data = pd.read_csv(filename)

label_related_columns = data.columns.isin(['class'])
predictors = np.logical_not(label_related_columns)

X = data.loc[:, predictors].to_numpy()
y = data[['class']].to_numpy()

print(X.shape)

X_train, X_test = X[::2], X[1::2]
y_train, y_test = y[::2], y[1::2]

#%%
clustering = KMeans(n_clusters=7, init = 'random', n_init = 10)
clustering.fit(X_train)
clusters = clustering.labels_

#%%
print('ARI on train', adjusted_rand_score(y_train.ravel(), clusters.ravel()))
print('ARI on test', adjusted_rand_score(clustering.predict(X_test).ravel(), y_test.ravel()))

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_train_reduced_by_PCA = pca.fit_transform(X_train)
X_test_reduced_by_PCA = pca.transform(X_test)
clustering = KMeans(n_clusters=7, init = 'random', n_init = 10)
clustering.fit(X_train_reduced_by_PCA)
clusters = clustering.labels_
print('ARI after PCA on train', adjusted_rand_score(y_train.ravel(), clusters.ravel()))
print('ARI after PCA on test', adjusted_rand_score(clustering.predict(X_test_reduced_by_PCA).ravel(), y_test.ravel()))
