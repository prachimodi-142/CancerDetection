#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import utils, models,datasets
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import time
import torch.nn.functional as F
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M


# In[2]:


class block(nn.Module):
    def __init__(self, in_planes, intermediate_planes, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = P4MConvP4M(in_planes, intermediate_planes, kernel_size=1, bias=False)
        self.conv2 = P4MConvP4M(intermediate_planes, intermediate_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = P4MConvP4M(intermediate_planes,intermediate_planes * self.expansion,kernel_size=1,stride=1,padding=0,bias=False)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        
        x = self.relu(x)
        x = self.conv2(x)
        
        x = self.relu(x)
        x = self.conv3(x)
        

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 23
        self.conv1 = P4MConvZ2(3, 23, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], intermediate_plane=23, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_plane=45, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_plane=91, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_plane=181, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(181*8* 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_plane, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_planes != intermediate_plane * 4:
            identity_downsample = nn.Sequential(
                P4MConvP4M(self.in_planes,intermediate_plane * 4,kernel_size=1,stride=stride,bias=False)
                
            )

        layers.append(block(self.in_planes, intermediate_plane, identity_downsample, stride))

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_planes = intermediate_plane * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_planes, intermediate_plane))

        return nn.Sequential(*layers)


# In[3]:


train_transform = transforms.Compose([transforms.Resize((96,96)),
                                      transforms.ColorJitter(brightness=.5, saturation=.25,hue=.1, contrast=.5),
                                      transforms.RandomAffine(10, (0.05, 0.05)),
                                      transforms.RandomHorizontalFlip(.5),
                                      transforms.RandomVerticalFlip(.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.6716241, 0.48636872, 0.60884315],
                                                           [0.27210504, 0.31001145, 0.2918652])
        ])

test_transform = transforms.Compose([transforms.Resize((96,96)),        
            transforms.ToTensor(),
            transforms.Normalize([0.6716241, 0.48636872, 0.60884315],
                                 [0.27210504, 0.31001145, 0.2918652])
        ])


# In[4]:


train_data = datasets.ImageFolder(root="PCam/Pcam_Train/Pcam_Train",transform=train_transform)
len(train_data)


# In[5]:


valid_data = datasets.ImageFolder(root="PCam/Pcam_Test_192/Pcam_Test_192",transform=test_transform)
len(valid_data)


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[7]:


model = ResNet(block, [3, 4, 6, 3], 3, 2)
model = model.to(device)


# In[8]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True,num_workers=24)
test_loader = DataLoader(valid_data, batch_size=128, shuffle=True,num_workers=24)


# In[ ]:


start_time = time.time()     
    
train_losses = []
train_correct = []
test_correct = []
best_acc = 0.0
    
for epoch in range(20):
        trn_corr = 0
        tst_corr = 0
        loop = tqdm(train_loader)
                
        # Run the training batches
        for b, (X_train, y_train) in enumerate(loop):       
            
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
     
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())   
           
    
            train_losses.append(loss.item())
            train_correct.append(trn_corr)
        
        print(f'epoch: {epoch:2} Train accuracy: {train_correct[-1].item()*100/len(train_data):.3f}%')
        
        # Run the testing batches
        
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                
                X_test, y_test = X_test.to(device), y_test.to(device)
                
                # Apply the model
                
                y_val = model(X_test)              
                
                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()
                test_correct.append(tst_corr)               
        
        
        print(f'epoch: {epoch:2} Test accuracy: {test_correct[-1].item()*100/len(valid_data):.3f}%')
        test_acc = test_correct[-1].item()*100/len(valid_data)
        if test_acc>best_acc:
            torch.save(model.state_dict(), "resnet50_group_"+str(test_acc)+"_"+str(epoch)+" .pt")
            best_acc = test_acc
       
        print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed    
    
print(f'Final Test accuracy: {test_correct[-1].item()*100/len(valid_data):.3f}%')

