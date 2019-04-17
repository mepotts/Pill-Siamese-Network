#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 11:38:35 2019

@author: katepotts
"""


import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import operator
import os
from fnmatch import fnmatch
import time

def get_inventory_paths():
    root = '/Users/katepotts/Pill-Detection/pill-inventory/pills/test'
    pattern = "*.jpg"
    
    pathlist = []
    
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                pathlist.append(os.path.join(path, name))
    return pathlist            

def find_inventory_match(image_name,network):
    t0 = time.time()
    inventory_paths = get_inventory_paths()
    image_path = "/Users/katepotts/Pill-Detection/pill-cropped/"+image_name
    img = Image.open(image_path)
    preprocess = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
    x0 = preprocess(img)
    x0 = x0.unsqueeze(0)
    class0 = os.path.splitext(os.path.basename(image_path))[0]
    
    lst = []
    
    for i in inventory_paths:
        img = Image.open(i)
        preprocess = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
        x1 = preprocess(img)
        x1 = x1.unsqueeze(0)
        class1 = os.path.splitext(os.path.basename(i))[0]
        concatenated = torch.cat((x0,x1),0)
        #print(class0, class1)
        net = network
        output1,output2 = net(Variable(x0),Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        lst.append([class0, class1, concatenated, euclidean_distance.item()])
    
    lst = sorted(lst, key=operator.itemgetter(3))
    print(lst[0][0], lst[0][1])
    imshow(torchvision.utils.make_grid(lst[0][2]),'Dissimilarity: {:.2f}'.format(lst[0][3]))
    t1 = time.time()
    print("Total inference time: "+str(t1-t0)+"s")

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    
class Config():
    training_dir = "./data-cropped/pills/training/"
    testing_dir = "./pill-inventory/pills/test/"
    train_batch_size = 16
    train_number_epochs = 1
    
class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        #img0 = img0.convert("L")
        #img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)), torch.from_numpy(np.array([int(img0_tuple[1])],dtype=np.float32)),torch.from_numpy(np.array([int(img1_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(12, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),


            nn.ReflectionPad2d(1),
            nn.Conv2d(12, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(24),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(24*100*100, 20),
            nn.ReLU(inplace=True),

            nn.Linear(20, 20),
            nn.ReLU(inplace=True),

            nn.Linear(20, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
