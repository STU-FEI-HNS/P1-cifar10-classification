# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:20:18 2022

@author: user
"""

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
#img size 32

class CustomCNN3(torch.nn.Module):
    def __init__(self):
        super(CustomCNN3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, padding=1),#32x32
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=2, stride=2),#16x16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, padding=1),#16x16
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2),#8x8
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=3, padding = 1),#8x8
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2),#4x4
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(16*20, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 10),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out

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
   'save_path':'my_serial_net',
}

root_train = 'train'
root_test = 'test'

NUM_TRAIN_IMAGES = 800
NUM_TEST_IMAGES = 200

transformation = transforms.Compose([ 
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation((0, 360), center=None),                                              
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

model = CustomCNN3()
criterion = torch.nn.CrossEntropyLoss()

   

optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

from torch.utils.tensorboard import SummaryWriter

my_net = CnnNet(model, params, trainloader, testloader, device)
my_net.train(criterion, optimizer)
my_net.test()
my_net.printResults()











