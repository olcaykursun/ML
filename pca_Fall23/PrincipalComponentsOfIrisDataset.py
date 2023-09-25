#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:59:29 2023

@author: okursun
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data  
y = iris.target
print("There are %d examples and %d features in the dataset" % (len(y), len(X[0])))

print("Features are:", iris.feature_names)
#There are 6 pairs of the 4 features, scatter plot the first two features as color-coded by class 
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width');


#%%
#Lets do PCA using the scikit library 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)       #Suppose we know we want 2 features, more on this in the next code cell...
X_reduced = pca.fit_transform(X)
plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k', s=40)
plt.xlabel("1st eigenvector")
plt.ylabel("2nd eigenvector")

#Make a new test example, lets just pick one example from the dataset or pick the center of mass of class-1
test_example=X[0:50,:].mean(axis=0)
projection_of_test_example = pca.transform([test_example])[0] #take the first list of the output
print(projection_of_test_example)
plt.scatter(projection_of_test_example[0], projection_of_test_example[1], c='b', marker='o', s=40)

print('Variance of the first component: ',np.var(np.dot(X, pca.components_[0]), ddof=1))

#%%
#Let do PCA using the mathematical formulation as in Chapter 6 (Dimensionality Reduction) of the book
#https://www.cmpe.boun.edu.tr/~ethem/i2ml3e/3e_v1-0/i2ml3e-chap6.pdf
means=X.mean(axis=0)
C=np.cov(X.T, ddof = 1)
print('C=', C)
evals,evects = np.linalg.eig(C)
print('eigen values = ',evals)
print(np.cumsum(evals)/sum(evals))

nfeats=min(np.where(np.cumsum(evals)/sum(evals) > 0.95)[0])+1
print('Number of features that explain 95% the variance:', nfeats)

projs=np.dot(X-means,evects[:,:nfeats])    #pca.transform does exactly this

myprojection_of_test_example=np.dot(test_example-means,evects[:,:nfeats])
print(myprojection_of_test_example)

plt.scatter(projs[:, 0], projs[:, 1], c=y, edgecolor='k', s=40)
plt.xlabel("1st eigenvector")
plt.ylabel("2nd eigenvector")
plt.scatter(myprojection_of_test_example[0], myprojection_of_test_example[1], c='b', marker='x', s=40)
