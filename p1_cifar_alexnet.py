# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:33:12 2022

@author: user
"""

import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import models, datasets
import torchvision.transforms as transforms
#from tqdm.notebook import trange, tqdm
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import random
from utils import CnnNet

print(f"Is GPU available? {torch.cuda.is_available()}")
print(f"Number of available devices: {torch.cuda.device_count()}")
print(f"Index of current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

params = {
    "bsize" : 200,# Batch size during training.
    'nepochs' : 20,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
   'freeze_first_n_layers' : 2,
   'save_path':'alexnet',
}

root_train = 'train'
root_test = 'test'

NUM_TRAIN_IMAGES = 800
NUM_TEST_IMAGES = 200

transformation = transforms.Compose([ 
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation((0, 360), center=None),    
            transforms.Resize(256),                                          
            transforms.ToTensor(),

    ])

train_dataset = datasets.ImageFolder(root=root_train, transform=transformation)
test_dataset = datasets.ImageFolder(root=root_test, transform=transformation)


# -------------- Odstranenie nadbytocnych vzoriek trenovacie data -----------------------
random.shuffle(train_dataset.samples) #nahodne zamiesanie
number_of_samples = [0]*len(train_dataset.classes)
to_be_removed = []
print("Trenovacie data")
print("Povodny pocet vzoriek: " + str(len(train_dataset.samples))) 

for path, class_num in train_dataset.samples:  #hladanie nadbytocnych vzoriek
  if (number_of_samples[class_num] >= NUM_TRAIN_IMAGES):
    to_be_removed.append((path,class_num))
  else:
    number_of_samples[class_num] += 1

for path, class_num in to_be_removed: #odstranovanie nadbytocnych vzoriek
  train_dataset.samples.remove((path,class_num))

print("Pocet vzoriek po odstranovani: " + str(len(train_dataset.samples)) + "\n")


# -------------- Odstranenie nadbytocnych vzoriek testovacie data -----------------------
random.shuffle(test_dataset.samples) #nahodne zamiesanie
number_of_samples = [0]*len(test_dataset.classes)
to_be_removed = []
print("Testovacie data")
print("Povodny pocet vzoriek: " + str(len(test_dataset.samples))) 

for path, class_num in test_dataset.samples:  #hladanie nadbytocnych vzoriek
  if (number_of_samples[class_num] >= NUM_TEST_IMAGES):
    to_be_removed.append((path,class_num))
  else:
    number_of_samples[class_num] += 1

for path, class_num in to_be_removed: #odstranovanie nadbytocnych vzoriek
  test_dataset.samples.remove((path,class_num))

print("Pocet vzoriek po odstranovani: " + str(len(test_dataset.samples)))

trainloader = DataLoader(dataset=train_dataset, batch_size=params['bsize'], shuffle=True)
testloader = DataLoader(dataset=test_dataset, batch_size=params['bsize'], shuffle=False)

model = models.alexnet(pretrained=True)
criterion = torch.nn.CrossEntropyLoss()

count_x = 0
count_y = 0
freeze_first_n_layers = params['freeze_first_n_layers']
# freeze backbone layers

for layer in model.children():
    if count_y < 1:
        count_y +=1
        for sublayer in layer.children():
            if count_x < freeze_first_n_layers and len(list(sublayer.parameters())) > 0: # freezing first 3 layers
                print(sublayer)
                sublayer.requires_grad_(False)
                count_x +=1   
      

optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

from torch.utils.tensorboard import SummaryWriter


model.classifier[6] = nn.Linear(4096, 10)

my_net = CnnNet(model, params, trainloader, testloader, device)
my_net.train(criterion, optimizer)


my_net.test()
my_net.printResults()