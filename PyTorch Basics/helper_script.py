import torch #main torch module
import torch.nn as nn #neural net module
import torch.optim as optim #optimizers
import torch.nn.functional as F #functions like ReLu Sig Tanh etc
from torch.utils.data import DataLoader #help us with datasets

import torchvision
import torchvision.datasets as datasets #using to access std data
import torchvision.transforms as transforms #transformations


def check_accuracy_FCN(loader,model):
    if loader.dataset.train:
        print("Checking Training Data Accuracy")
    else:
        print("Checking Test Data Accuract")
    
    num_correct = 0
    num_samples = 0
    model.eval() #set to evaluation mode
    
    with torch.no_grad():
        #only have to check accuracy, dont compute grads
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0],-1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        accuracy = float(num_correct)/float(num_samples)*100
        print(f"Got {num_correct} / {num_samples} with accuracy {accuracy: .2f}")

    model.train()
    return accuracy

def check_accuracy_CNN(loader,model):
    if loader.dataset.train:
        print("Checking Training Data Accuracy")
    else:
        print("Checking Test Data Accuracy")
    
    num_correct = 0
    num_samples = 0
    model.eval() #set to evaluation mode
    
    with torch.no_grad():
        #only have to check accuracy, dont compute grads
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            #x = x.reshape(x.shape[0],-1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        accuracy = float(num_correct)/float(num_samples)*100
        print(f"Got {num_correct} / {num_samples} with accuracy {accuracy: .2f}")

    model.train()
    return accuracy

def epoch_accuracy_CNN(loader,model):
    #either train or test accuracy
    num_correct = 0
    num_samples = 0
    model.eval() #set to evaluation mode
    with torch.no_grad():
        #only have to check accuracy, dont compute grads
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            #x = x.reshape(x.shape[0],-1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    accuracy = float(num_correct)/float(num_samples)
    model.train()
    return accuracy 

def save_checkpoint_CNN(state, filename="./checkpoints/my_checkpoint.pth.tar"):
    print("Saving Checkpoint ==>")
    torch.save(state, filename)
    
def load_checkpoint_CNN(checkpoint):
    print("Loading Checkpoint ==>")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])