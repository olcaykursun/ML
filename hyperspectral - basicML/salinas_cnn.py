#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 21:51:43 2022

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
print(f'Double check the ground-truth shape: {gt.shape=}')

gt_array = gt.reshape((num_rows*num_cols,))
class_names = sorted(list(set(gt_array)))
print('Classes in GT', class_names)

from collections import Counter
counts2 = Counter(gt_array)
print([(i, counts2[i]) for i in sorted(counts2.keys())])

#%%
from sklearn.model_selection import train_test_split
data_array = data.reshape((num_rows*num_cols, -1))

# labeled_pixels = (gt_array > 0)
# class_names.remove(0)
# data_array = data_array[labeled_pixels]
# gt_array = gt_array[labeled_pixels]

gt_array[gt_array>0] = 1

X_train, X_test, y_train, y_test = train_test_split(data_array, gt_array, test_size=0.9, random_state=123, stratify=gt_array)

import matplotlib.pyplot as plt
plt.plot(X_train.var(axis=0))

#%%
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('NB:', (y_test == y_pred).mean())
print(y_test.mean())

#%%

import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, pixels, labels):
        self.X = pixels
        self.y = labels

    def __getitem__(self, index):
        pixel = self.X[index]
        pixel = np.expand_dims(pixel, axis=0)
        pixel = np.expand_dims(pixel, axis=0)
        pixel = np.expand_dims(pixel, axis=0)
        pixel = torch.tensor(pixel)
        return pixel, self.y[index]

    def __len__(self):
        return len(self.X)

batch_size = 12
train_dataset = MyDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



#%%
import torch.nn as nn
import torch.nn.functional as F

num_output = 1 #binary classification (probability of class1)
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 16, (1, 1, 11), stride=(1,1,4))
        self.pool1 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2))
        self.conv2 = nn.Conv3d(16, 32, (1, 1, 5), stride=(1,1,2))
        self.pool2 = nn.AdaptiveMaxPool3d((1,1,2))        
        self.fc = nn.Linear(32*2, num_output)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

net = Net()
net = net.float()


#%%
import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()
net.train()

optimizer = optim.Adam(net.parameters(), lr=1e-3)
Nepoch = 50
    
for epoch in range(Nepoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels.view(outputs.shape).type_as(outputs))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'[{epoch + 1}] loss: {running_loss:.3f}')
        

#%%
test_dataset = MyDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_correct = 0
for data in test_loader:
    inputs, labels = data
    outputs = net(inputs.float())
#    preds = (outputs.flatten() > 0) + 0
    preds = torch.gt(outputs.flatten(), 0).int()
    num_correct += torch.sum(preds == labels).item()

print(num_correct / len(X_test))
