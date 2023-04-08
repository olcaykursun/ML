#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 21:02:12 2022

@author: okursun
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def plot_data2D(data,labels):
  x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
  y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5

  fig = plt.figure(0, figsize=(8, 6))
  plt.clf()

  # Plot the training points
  plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Set1, edgecolor='k')
  plt.xlabel('x1')
  plt.ylabel('x2')

  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
  
  plt.show()
  return fig

def plot_linear_discriminant(fig, w):
  xx = np.linspace(-5, 5)
  yy=  (-w[0]-xx*w[1])/w[2]
  plt.figure(fig)
  plt.plot(xx, yy, 'k-')
  plt.show()
  
from math import exp
def sigmoid(a):
  return 1 / (1 + exp(-a))


n_samples=100
d = 2   #number of dimensions
cluster_std=0.1
mean0 = [0, 0]
mean1 = [1, 0]
mean2 = [1, 1]
mean3 = [0, 1]  
pos_class = 2    #which center is positive?


from sklearn.datasets import make_blobs
data, target = make_blobs(n_samples=n_samples, n_features=2, 
      centers=[mean0, mean1, mean2, mean3], cluster_std=cluster_std, random_state=1)


pos_examples = target == pos_class
neg_examples = target != pos_class
target[pos_examples] = 1
target[neg_examples] = 0

# dataset = np.c_[data, target]
# dataset = np.hstack((data,target.reshape(-1,1)))
# dataset = np.concatenate((data,target.reshape(-1,1)),axis=1)

print(f'data is {data.shape}')
#print(f'{data.shape=}')
print(f'target is {target.shape}')

fig1 = plot_data2D(data, target);
init_randomly = False


#linear perceptron
Nepochs = 100
eta=0.01    #learning rate

train_x = np.insert(data, 0, 1, axis=1)

if init_randomly:
    rng = np.random.RandomState(1234567)
    w = rng.randn(d+1)/5           #1D is sufficient until we add a hidden layer with hidden units
else:
    w = np.array([1, 1, -1])
    plot_linear_discriminant(fig1, w)
time.sleep(1)
    
    
for epoch in range(Nepochs+2):    #Times to go over the data (no updates in the first and last epochs to see accuracies)
  numcorr=0
  E=0
  for t in range(n_samples):
    r=target[t]
    x=train_x[t,:]
    y=sigmoid(np.dot(x,w))
    if (y>0.5 and r==1) or (y<=0.5 and r==0) :
      numcorr=numcorr+1     #this calculates the training accuracy
    E = E - (r*np.log2(y) + (1-r)*np.log2(1-y))
    delta=r-y
    if epoch not in [0, Nepochs+1]:
      w=w+eta*delta*x       #update
  print(epoch, numcorr/n_samples, E/n_samples)            #Error is expected reduce with updates
  if epoch % 5 == 4:
    plot_linear_discriminant(fig1, w)
    time.sleep(0.5)



