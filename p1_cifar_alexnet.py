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
from utils import save_plots, SaveBestModel



def printGrph(params, loss_history, acc_history):
    plt.plot(np.array(range(1, params['nepochs']  + 1)), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(np.array(range(1, params['nepochs']  + 1)), acc_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

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
   'freeze_first_n_layers' : 4,
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

print(model)

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

writer = SummaryWriter('runs/cifar10_' + params['save_path'])

model.classifier[6].out_features = 10
print(model)

model = model.to(device)

model.train() #activate training mode

acc_history = []
loss_history = []
final_labels = []
final_predicted = []

save_best_model = SaveBestModel(path=params['save_path'])

for epoch in trange(1, params['nepochs'] + 1, desc="1st loop"):
    epoch_loss = 0
    n_batches = len(train_dataset) // params['bsize']
    correct = 0
    total = 0
    accuracy_train = 0

    for step, (images, labels) in enumerate(tqdm(trainloader, desc="Epoch {}/{}".format(epoch, params['nepochs']))):

        images = images.to(device)
        labels = labels.to(device)
        
        # Dopredne sirenie, 
        # ziskame pravdepodobnosti tried tym, ze posleme do modelu vstupy
        outputs = model(images)

        # Vypocitame chybu algoritmu       
        loss = criterion(outputs, labels)
        
        # Uspesnost algoritmu
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy_train = correct / total
        epoch_loss += loss.item() 
        
        # Je vhodne zavolat zero_grad() pred zavolanim spatneho sirenia 
        # pre vynulovanie gradientov z predosleho volania loss.backward()
        optimizer.zero_grad()

        # Spatne sirenie chyby, vypocitaju sa gradienty
        loss.backward()
        
        # Aktualizacia vah pomocou optimalizatora
        optimizer.step()
    
        if (step+1) % n_batches == 0 and step != 0:
            print(str(step))
            epoch_loss = epoch_loss / n_batches

            writer.add_scalar(
                'Training loss',
                epoch_loss,
                epoch
            )
            writer.add_scalar(
                'Trainning accuracy',
                accuracy_train,
                epoch
            )

            acc_history.append(accuracy_train)
            loss_history.append(epoch_loss)
            print("Epoch {}, Loss {:.6f}, Accuracy {:.2f}% ".format(epoch, epoch_loss, accuracy_train * 100))
            epoch_loss = 0

            #print(model.layer1[0].conv1.weight[0][0])
            #print(model.layer2[0].conv1.weight[0][0])
            #print(model.layer3[0].conv1.weight[0][0])
        save_best_model(epoch_loss, epoch, model, optimizer, criterion) #To save best model
        final_predicted += predicted.tolist()
        final_labels += labels.tolist()
        torch.cuda.empty_cache()



writer.add_hparams(
    {
    'optimizer': optimizer.__class__.__name__,
    'lr': params['lr'], 
    'batch_size': params['bsize']
    },
    {
    'hparam/train/accuracy': accuracy_train,
    }
)
writer.close()

model.eval()  # activate evaulation mode, some layers behave differently
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
total = 0
correct = 0
final_labels = []
final_predicted = []
for inputs, labels in tqdm(iter(testloader), desc="Full forward pass", total=len(testloader)):
    if use_cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
    with torch.no_grad():
        outputs_batch = model(inputs)

    _, predicted = torch.max(outputs_batch.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    final_predicted += predicted.tolist()
    final_labels += labels.tolist()

confm = confusion_matrix(final_labels, final_predicted)
print(confm)
print('Accuracy of the network on the test images: %0.2f %%' % (100 * correct / total))

writer.flush()

from pretty_confusion_matrix import pp_matrix_from_data

labels = [i for i in range(10)]

pp_matrix_from_data(final_labels, final_predicted, params['save_path'], columns=labels, cmap="gnuplot")
save_plots(acc_history, loss_history, params['save_path'])


