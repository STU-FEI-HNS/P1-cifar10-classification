# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:23:17 2022

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
    "bsize" : 50,# Batch size during training.
    'nepochs' : 20,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
   'freeze_first_n_layers' : 4,
   'save_path':'ensemblenet',
}

root_train = 'train'
root_test = 'test'

NUM_TRAIN_IMAGES = 800
NUM_TEST_IMAGES = 200

transformation = transforms.Compose([ 
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation((0, 360), center=None),    
            # transforms.Resize(256),                                          
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

#----------------------------model1------------------------------
model1 = models.resnet18(pretrained=True)
model1.fc = nn.Linear(512,10) 
# model1.fc.out_features = 10
criterion = torch.nn.CrossEntropyLoss()
# weights = torch.load('weights/resnet_final_model.pth')
# model1.load_state_dict(weights["model_state_dict"])
my_net1 = CnnNet(model1, params, trainloader, testloader, device)


# my_net1.loadWeights('weights/resnet_final_model.pth')


my_net1.model.load_state_dict(torch.load('resnet_final_model1.pth'))
result1,trueclasses = my_net1.test2()
# my_net1.printResults()

#---------------------------model2-----------------------------
model2 = models.vgg16(pretrained=True)
model2.classifier[6] = nn.Dropout(0.2) #pridanie dropout vrstvy
model2.classifier.append(nn.Linear(4096,10))

my_net2 = CnnNet(model2, params, trainloader, testloader, device)
my_net2.loadWeights('weights/vggnet_final_model.pth')
result2,_ = my_net2.test2()
# my_net2.printResults()

#---------------------------model3-----------------------------
model3 = models.googlenet(pretrained=True)
#model3.classifier[6] = nn.Linear(4096, 10)
model3.fc.out_features = 10 

my_net3 = CnnNet(model3, params, trainloader, testloader, device)
my_net3.loadWeights('weights/googlenet_final_model.pth')
result3,_ = my_net3.test2()
# my_net3.printResults()

#----------------------------Ensemble learning---------------------

#normalizacia na rozsah 0-1
result1 = np.divide(result1,np.amax(result1,axis=1).reshape(-1,1))
result2 = np.divide(result1,np.amax(result2,axis=1).reshape(-1,1))
result3 = np.divide(result1,np.amax(result3,axis=1).reshape(-1,1))
result1 = result1.clip(min=0)
result2 = result1.clip(min=0)
result3 = result1.clip(min=0)


resultsoft = np.add(result1,result2,result3) #spocitanie vah (soft voting)
softvoting = np.argmax(resultsoft,axis=1)

result1 = np.floor(result1)
result2 = np.floor(result2)
result3 = np.floor(result3)
# print(result1)

resulthard = np.add(result1,result2,result3) #spocitanie vah (hard voting)
hardvoting = np.argmax(resulthard,axis=1)

from pretty_confusion_matrix import pp_matrix_from_data
labels = [i for i in range(10)]
pp_matrix_from_data(trueclasses, softvoting, 'Soft_voting', columns=labels, cmap="gnuplot")
pp_matrix_from_data(trueclasses, hardvoting, 'Hard_voting', columns=labels, cmap="gnuplot") 