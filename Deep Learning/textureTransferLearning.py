#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 21:02:36 2023

@author: okursun
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as tmodels
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torchvision.models import AlexNet_Weights

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

Nwin = 60     #snip off patches (or examples) of 30x30 pixel field 
N_files = 13
N_classmember_TR = 10    # few examples per class for training
N_classmember_TS = 20    # some for testing
N_TR = N_classmember_TR * N_files    #Few examples to learn 11x11x3 = 363 weights from scratch in conv1
N_TS = N_classmember_TS * N_files    #unless you load a pretrained network
data_dir = './'
ftype = '.tiff'

im_transform = transforms.Compose([
 transforms.ToPILImage(),
 transforms.Grayscale(3),
 transforms.RandomCrop(Nwin),
 transforms.ToTensor(),             
 transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]     
 )
])

class MyDataset(Dataset):
    def __init__(self, images, nclassmember, transform=None):
        super(MyDataset, self).__init__()
        self.images = images
        self.transform = transform
        self.nclassmember = nclassmember
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return {'X':img,'Y':idx//self.nclassmember}


def newtask():
    trimages = []
    tsimages = []
    for idx in range(N_files):
        img_name = os.path.join(data_dir,"1.2."+str(idx+1).zfill(2)+ftype)
        im = io.imread(img_name)
        for j in range(N_classmember_TR):
            trimages.append(im_transform(im))
        for j in range(N_classmember_TS):
            tsimages.append(im_transform(im))
    dataset_TR = MyDataset(trimages, N_classmember_TR, transform = None) #Use transform here for data augmentation 
    dataset_TS = MyDataset(tsimages, N_classmember_TS, transform = None) #Dont use im_transform here
    a=DataLoader(dataset_TR, batch_size=N_TR, shuffle=True, num_workers=0, pin_memory=True)
    b=DataLoader(dataset_TS, batch_size=N_TS, shuffle=True, num_workers=0, pin_memory=True) 
    return a, b

def plot_filters_multi_channel(t):
    t=t.cpu()
    num_kernels = t.shape[0]    
    num_cols = 8
    num_rows = num_kernels
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        npimg = np.array(t[i].numpy(), np.float32)
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        if npimg.shape[2]==1:
            ax1.imshow(np.concatenate((npimg,npimg,npimg),axis=2))
        else:
            ax1.imshow(npimg)
        ax1.set_xticks([])
        ax1.set_yticks([])
    
class TransferedBase(nn.Module):
            def __init__(self, children, numlayers):
                super(TransferedBase, self).__init__()
                self.features = nn.Sequential(
                    *list(children())[:numlayers],
                    nn.AdaptiveAvgPool2d((1,1))
                )

                #For feature extraction, this final classifier layer and "forward" is not needed
                #Needed only if the network is retrained      
#               self.fc = nn.Linear(64,N_files)   #depends on which layer to use in the transfer
                                
            def forward(self, x):
#                x = self.features(x)
#                x = x.view(-1, 64)
#                x = self.fc(x)
                return x

#%%
net = tmodels.alexnet(weights=AlexNet_Weights.DEFAULT)
plot_filters_multi_channel(net.features[0].weight.data)
print(net)

num_layers_transferred = 3    #Use 3 layers for Area-1,  use 6 layers for Area-2, ...
baseNet = TransferedBase(net.features.children, num_layers_transferred).to(device)  
print(baseNet)

baseNet.eval()   #Let's not retrain the network, let's simply use it for feature extraction.

#%%    
trainloader, testloader = newtask()

dataiter = iter(trainloader)
train_batch = next(dataiter)
raw_data = train_batch['X'].to(device)
features = baseNet.features(raw_data).detach().cpu().numpy().reshape((N_TR, -1))

clf2 = LogisticRegression(max_iter=5000)  #or use KNN or Random Forest
clf2.fit(features, train_batch['Y'].numpy())

dataiter = iter(testloader)
test_batch = next(dataiter)
raw_dataTS = test_batch['X'].to(device)
featuresTS = baseNet.features(raw_dataTS).detach().cpu().numpy().reshape((N_TS, -1))
print(clf2.score(featuresTS, test_batch['Y'].numpy()))
