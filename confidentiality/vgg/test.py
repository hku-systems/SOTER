import torch
import time
import os
import copy
import math
import models7a

import numpy as np
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image

batch_size = 64

PATH13S10 = "../data/standard_vgg13_50%.pkl"
PATH19S10 = "../data/standard_vgg19_77%.pkl"
PATH13S100 = "../data/standard_vgg13-100-2800.pkl"
PATH19S100 = "../data/standard_vgg19-100-18000.pkl"
PATH_new  = "../data/ini.pkl"
#######################################################################################33
###   data loading   ###

def load_data(Size=50000, kind="cifar-10"):
    if kind == "cifar-10":
        train_dataset = datasets.CIFAR10(root='../data',train=True,transform=transforms.ToTensor(),download=True)
        test_dataset  = datasets.CIFAR10(root='../data',train=False, transform=transforms.ToTensor())

    elif kind == "cifar-100":
        train_dataset = datasets.CIFAR100(root='../data',train=True,transform=transforms.ToTensor(),download=True)
        test_dataset  = datasets.CIFAR100(root='../data', train=False,transform=transforms.ToTensor())

    train_dataset_p = models7a.MyDataset(Dataset=train_dataset,size=Size)
    train_loader = DataLoader(dataset=train_dataset_p,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False)
        
    return train_loader,test_loader
#######################################################################


def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.

    for i, (features, targets) in enumerate(data_loader):  
        features = features.to(device)
        targets = targets.to(device)
        outputs = model(features)
        probas = F.softmax(outputs,dim = 1)
 
        cross_entropy += F.cross_entropy(outputs, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100, cross_entropy/num_examples

##############################################################################################
### test  ###
def test(PATH,model,DEVICE,test_loader):
    weights_origin=torch.load(PATH)
    model.load_state_dict(weights_origin)
    model.eval()
    model.to(DEVICE)

    with torch.set_grad_enabled(False): # save memory during inference
        test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, DEVICE)
        print(f'Test accuracy: {test_acc:.8f}%')


###################################################################################

def main(modelname,p,num_classes,freezeornot):
    
    if num_classes==10:
        PATH_PARA = PATH13S10
        if not modelname=="ennclave":
            PATH_PARA = PATH19S10
    elif num_classes==100:
        PATH_PARA = PATH13S100
        if not modelname=="ennclave":
            PATH_PARA = PATH19S100


    if freezeornot:
        model = models7a.vgg13_freeze(num_classes=num_classes,partition=p)
    else:
        model = models7a.vgg13_freeze(num_classes=num_classes)
    


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH="../data/"+modelname+"-"+str(p)+"-"+str(freezeornot)+".pkl"
    if modelname=="standard-line":
        PATH=PATH19S100
        model = models7a.vgg19(num_classes=num_classes)
    elif modelname=="mlcapsule":
        PATH=PATH13S100

    train_loader,test_loader = load_data()
    print(modelname,p)
    test(PATH,model,DEVICE,test_loader)

if __name__=='__main__':  

    cifar=100
    frz=False
    partitions=[0,20,40,60,80,100]
    models    =["ennclave","soter","aegisdnn"]

    main("mlcapsule",0,cifar,frz)
    main("standard-line",0,cifar,frz)
    for model in models:
        for partition in partitions:
            main(model,partition,cifar,frz)


################################################## ends #####################################
