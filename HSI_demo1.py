#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:50:50 2023

@author: okursun
"""

from scipy.io import loadmat
from matplotlib import pyplot as plt

dataDir = '/Users/okursun/Downloads/pines/'
#https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
data = loadmat(dataDir+'Salinas_corrected.mat')['salinas_corrected']

gt = loadmat(dataDir+'Salinas_gt.mat')['salinas_gt']

plt.axis("off")
plt.imshow(gt)
plt.show()

#%%
plt.axis("off")
plt.imshow(data[:,:,5])
plt.show()

#%%
num_rows = data.shape[0]
num_cols = data.shape[1]
num_bands = data.shape[2]

print(f'Image Size: {(num_rows, num_cols)}\nNumber of Bands: {num_bands}')

gt_array = gt.reshape((num_rows*num_cols,))
class_names = sorted(list(set(gt_array)))
print('Classes in GT', class_names)

plt.hist(gt_array,30)

#%%
list_gt = list(gt_array)
class_counts = {i: list_gt.count(i) for i in sorted(class_names)}

#%%
from sklearn.model_selection import train_test_split
data_array = data.reshape((num_rows*num_cols, -1))
#%%

labeled_pixels = (gt_array > 0)
class_names.remove(0)

data_array = data_array[labeled_pixels]
gt_array = gt_array[labeled_pixels]

X_train, X_test, y_train, y_test = train_test_split(data_array, gt_array, test_size=0.9, random_state=123, stratify=gt_array)

#%%
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
y_pred = nb_clf.predict(X_test)
print('Naive Bayes:', (y_test == y_pred).mean())

from sklearn.neighbors import NearestCentroid
nc_clf = NearestCentroid()
nc_clf.fit(X_train, y_train)
print('Nearest Centroid:', nc_clf.score(X_test, y_test))

#%%
fig, ax = plt.subplots()
ax.plot(X_train.min(axis=0))
ax.plot(X_train.max(axis=0))
plt.show()

plt.plot(X_train.std(axis=0))

#%%
from sklearn.preprocessing import StandardScaler
#z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
nb_clf.fit(X_train_scaled, y_train)
print('Naive Bayes:', nb_clf.score(X_test_scaled, y_test))

nc_clf.fit(X_train_scaled, y_train)
print('Nearest Centroid:', nc_clf.score(X_test_scaled, y_test))

#%%
#https://machinelearningmastery.com/dont-use-random-guessing-as-your-baseline-classifier/
class_counts.pop(0)
dummy = max(class_counts.values()) / sum(class_counts.values())
print('ZeroR:', dummy)

#But I would use the training set (especially if the train-test split was not stratified)
from scipy import stats
class_mode = stats.mode(y_train).mode[0]
dummy = (y_train == class_mode).mean()
print('ZeroR:', dummy)

#%%
#Making sure we process train and test similarly
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('clf', NearestCentroid())])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
