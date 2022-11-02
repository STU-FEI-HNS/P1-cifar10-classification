# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:52:42 2022

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

from torchvision.transforms.autoaugment import AutoAugmentPolicy


print(f"Is GPU available? {torch.cuda.is_available()}")
print(f"Number of available devices: {torch.cuda.device_count()}")
print(f"Index of current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)

params = {
    "bsize" : 100,# Batch size during training.
    'nepochs' : 20,# Number of training epochs.
    'lr' : 0.0004,# Learning rate for optimizers
   'freeze_first_n_layers' : 3,
   'save_path':'googlenet_aug_off',
}

root_train = 'train'
root_test = 'test'

NUM_TRAIN_IMAGES = 100
NUM_TEST_IMAGES = 200

transformation = transforms.Compose([ 
            transforms.Resize(256),                                          
            transforms.ToTensor(),

    ])
transformation_test = transforms.Compose([    
            transforms.Resize(256),                                          
            transforms.ToTensor(),
    ])

train_dataset = datasets.ImageFolder(root=root_train, transform=transformation)
test_dataset = datasets.ImageFolder(root=root_test, transform=transformation_test)


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


model = models.googlenet(pretrained=True)
criterion = torch.nn.CrossEntropyLoss()

print(model)

count = 0
freeze_first_n_layers = params['freeze_first_n_layers']
# freeze backbone layers
for param in model.children(): 
    if count < freeze_first_n_layers and len(list(param.parameters())) > 0: # freezing first 3 layers
        print(param)
        param.requires_grad_(False)
        count +=1   
        
optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])

from torch.utils.tensorboard import SummaryWriter

model = models.googlenet(pretrained=True)
model.fc.out_features = 10

my_net = CnnNet(model, params, trainloader, testloader, device)
my_net.loadWeights('weights/googlenet_final_model.pth')

#------------------Zobrazenie filtrov pre Conv1, Conv2, Conv3---------------------
from torchvision import utils, models
import matplotlib.pyplot as plt

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
  n,c,w,h = tensor.shape

  if allkernels: tensor = tensor.view(n*c, -1, w, h)
  elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

  rows = np.min((tensor.shape[0] // nrow + 1, 64))    
  grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
  plt.figure( figsize=(nrow,rows) )
  plt.imshow(grid.numpy().transpose((1, 2, 0)))

filter1 = my_net.model.conv1.conv.weight.data.clone().cpu()
filter2 = my_net.model.conv2.conv.weight.data.clone().cpu()
filter3 = my_net.model.conv3.conv.weight.data.clone().cpu()

visTensor(filter1, ch=0, allkernels=False)
visTensor(filter2, ch=0, allkernels=False)
visTensor(filter3, ch=0, allkernels=False)


#------------------Pokus o zobrazenie feature map---------------------
#Pre test som to ribil pre conv1 ale asi to treba spraviť pre nejakú 
#hlbšiu vrstvulebo v zadaní je že výstupnú mapu príznakoc


model_weights =  my_net.model.conv1.conv.weight

images = next(iter(trainloader))
#images = images.cuda()
image_b = images[0]
image = image_b[0]

image = image.to(device)
outputs = []
names = []

image = my_net.model.conv1.conv(image)
image2 = my_net.model.conv2.conv(image)
image3 = my_net.model.conv3.conv(image2)

print(len(image))

image = image.squeeze(0)
gray_scale = torch.sum(image,0)
gray_scale = gray_scale / image.shape[0]
image_new = gray_scale.data.cpu().numpy()
fig = plt.figure(figsize=(32, 32))
imgplot = plt.imshow(image_new)

image2 = image2.squeeze(0)
gray_scale = torch.sum(image2,0)
gray_scale = gray_scale / image2.shape[0]
image2_new = gray_scale.data.cpu().numpy()
fig = plt.figure(figsize=(32, 32))
imgplot = plt.imshow(image2_new)


image3 = image3.squeeze(0)
gray_scale = torch.sum(image3,0)
gray_scale = gray_scale / image3.shape[0]
image3_new = gray_scale.data.cpu().numpy()
fig = plt.figure(figsize=(32, 32))
imgplot = plt.imshow(image3_new)

plt.axis("off")
plt.show()
