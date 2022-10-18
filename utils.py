# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:43:37 2022

@author: user
"""
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter



plt.style.use('ggplot')

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, path, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.path = path
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest train loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'weights/'+ self.path +'_final_model.pth')

def save_plots(train_acc, train_loss, path):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('figures/'+ path +'_accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/'+ path +'_loss.png')
    
class CnnNet:
    """
    Class for operate the neral network, mainly for training and etsting of model.
    """
    def __init__(
        self, model, params, trainloader, testloader, device):
        self.params = params
        self.model = model
        self.device = device
        self.writer = SummaryWriter('runs/cifar10_' + self.params['save_path'])
        self.acc_history = []
        self.loss_history = []
        self.final_labels = []
        self.final_predicted = []
        self.trainloader = trainloader
        self.testloader = testloader
        self.total = 0
        self.correct = 0
        self.save_best_model = SaveBestModel(path=self.params['save_path'])
        
  
    def train(self, criterion, optimizer):
        
        self.model = self.model.to(self.device)
        self.model.train() #activate training mode
        
        for epoch in trange(1, self.params['nepochs'] + 1, desc="1st loop"):
            epoch_loss = 0
            n_batches = len( self.trainloader.dataset) // self.params['bsize']
            correct = 0
            total = 0
            accuracy_train = 0

            for step, (images, labels) in enumerate(tqdm(self.trainloader, desc="Epoch {}/{}".format(epoch, self.params['nepochs']))):

                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Dopredne sirenie, 
                # ziskame pravdepodobnosti tried tym, ze posleme do modelu vstupy
                outputs = self.model(images)

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

                    self.writer.add_scalar(
                        'Training loss',
                        epoch_loss,
                        epoch
                    )
                    self.writer.add_scalar(
                        'Trainning accuracy',
                        accuracy_train,
                        epoch
                    )

                    self.acc_history.append(accuracy_train)
                    self.loss_history.append(epoch_loss)
                    print("Epoch {}, Loss {:.6f}, Accuracy {:.2f}% ".format(epoch, epoch_loss, accuracy_train * 100))
                    epoch_loss = 0

                
                self.save_best_model(epoch_loss, epoch, self.model, optimizer, criterion) #To save best model
                self.final_predicted += predicted.tolist()
                self.final_labels += labels.tolist()
                torch.cuda.empty_cache()

        self.writer.add_hparams(
            {
            'optimizer': optimizer.__class__.__name__,
            'lr': self.params['lr'], 
            'batch_size': self.params['bsize']
            },
            {
            'hparam/train/accuracy': accuracy_train,
            }
        )
        self.writer.close()
        
    
    def test(self):
        self.model.eval()  # activate evaulation mode, some layers behave differently
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        self.final_labels = []
        self.final_predicted = []
        for inputs, labels in tqdm(iter(self.testloader), desc="Full forward pass", total=len(self.testloader)):
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                outputs_batch = self.model(inputs)

            _, predicted = torch.max(outputs_batch.data, 1)
            self.total += labels.size(0)
            self.correct += (predicted == labels).sum().item()

            self.final_predicted += predicted.tolist()
            self.final_labels += labels.tolist()
        return self.model
    
    
    def printResults(self):
        confm = confusion_matrix(self.final_labels, self.final_predicted)
        print(confm)
        print('Accuracy of the network on the test images: %0.2f %%' % (100 * self.correct / self.total))

        self.writer.flush()

        from pretty_confusion_matrix import pp_matrix_from_data

        labels = [i for i in range(10)]

        pp_matrix_from_data(self.final_labels, self.final_predicted, self.params['save_path'], columns=labels, cmap="gnuplot")
        save_plots(self.acc_history, self.loss_history, self.params['save_path'])
    
    
    