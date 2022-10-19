#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 21:20:30 2022

@author: okursun
"""


#%%
from sklearn.model_selection import train_test_split
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123, stratify=y)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('NB:', (y_test == y_pred).mean())

#%%
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, pixels, labels):
        self.X = pixels
        self.y = labels

    def __getitem__(self, index):
        pixel = self.X[index]
        pixel = torch.tensor(pixel)
        return pixel, self.y[index]

    def __len__(self):
        return len(self.X)

batch_size = 16
train_dataset = MyDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#%%
import torch.nn as nn
import torch.nn.functional as F

num_output = 3
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 1)
        self.fc2 = nn.Linear(1, num_output)

    def forward(self, x):
        x = (self.fc1(x))
        #print('after the first layer', x.shape)
        x = self.fc2(x)
        #print('after the second layer', x.shape)
        return x

net = Net()
net = net.float()

#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
net.train()

optimizer = optim.Adam(net.parameters(), lr=.1)
Nepoch = 1000
    
for epoch in range(Nepoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 1):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'[{epoch}] loss: {running_loss:.3f}')
        
#%%
test_dataset = MyDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_correct = 0
for data in test_loader:
    inputs, labels = data
    outputs = F.softmax(net(inputs.float()), dim = 1)
    _, preds = torch.max(outputs, 1)
    num_correct += torch.sum(preds == labels).item()

print(num_correct / len(X_test))
