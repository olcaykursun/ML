#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 6 22:34:36 2023

@author: okursun
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy.stats import mode

digits = datasets.load_digits()
print(digits.data.shape)

#%%
plt.figure()
plt.gray()
for i in range(10):
    plt.matshow(digits.images[i])
    plt.axis('off')

#%%
fig, axes = plt.subplots(nrows=1, ncols=10)
plt.gray()
for i in range(10):
    ax = axes[i]
    ax.matshow(digits.images[i])
    ax.axis('off')    

#%%
fig, axes = plt.subplots(nrows=2, ncols=10)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.25,
                    wspace=0.1,
                    hspace=0.1)
plt.gray()
for i in range(10):
    ax = axes[0,i]
    ax.matshow(digits.images[i][:4])
    ax.axis('off')
    
    ax = axes[1,i]
    ax.matshow(digits.images[i][4:])
    ax.axis('off')

    ax.text(3, 7, str(digits.target[i]))

    
#%%
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.9, random_state=220, stratify=digits.target)

D = digits.data.shape[1]
ntest = len(y_test)

errors=[]
errors_knn_clf=[]
for n_neighbors in range(1,15):

  nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X_train)
  distances, indices = nbrs.kneighbors(X_test)

  error = 0
  for i in range(ntest):    
      error += mode(y_train[indices[i]]).mode[0] != y_test[i]
  error = error / ntest
  errors.append(error)
  
  knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
  knn_clf.fit(X_train, y_train)
  error_knn_clf = 1 - knn_clf.score(X_test, y_test)
  errors_knn_clf.append(error_knn_clf)

  print(error, error_knn_clf)


plt.figure()
plt.plot(errors, color='r', label='ours')
plt.plot(errors_knn_clf, color='b', label='scikit')

plt.xlabel("#Neighbors (k)")
plt.ylabel("Classification Error (%)")
plt.title("KNN Error")
  
# Adding legend 
plt.legend()
