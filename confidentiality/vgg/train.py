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

PATH13S10 = "../data2/standard_vgg13-10-3000.pkl"
PATH19S10 = "../data2/standard_vgg19-10-21000.pkl"
PATH13S100 = "../data2/standard_vgg13-100-2800.pkl"
PATH19S100 = "../data2/standard_vgg19-100-18000.pkl"
PATH_new  = "../data2/ini.pkl"
#######################################################################################33
###   data loading   ###

def load_data(Size=50000, kind="cifar-10"):
    if kind == "cifar-10":
        train_dataset = datasets.CIFAR10(root='../data2',train=True,transform=transforms.ToTensor(),download=True)
        test_dataset  = datasets.CIFAR10(root='../data2',train=False, transform=transforms.ToTensor())

    elif kind == "cifar-100":
        train_dataset = datasets.CIFAR100(root='../data2',train=True,transform=transforms.ToTensor(),download=True)
        test_dataset  = datasets.CIFAR100(root='../data2', train=False,transform=transforms.ToTensor())

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


######################################################################################
### training ###

def train(model,DEVICE,NUM_EPOCHS,train_loader,PATH,test_loader,printing=True,saving=True):
    train_acc_lst, valid_acc_lst = [], []
    train_loss_lst, valid_loss_lst = [], []
    start_time=time.time()

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    valid_loader = test_loader

    for epoch in range(NUM_EPOCHS):

        model.train()    
        for batch_idx, (features, targets) in enumerate(train_loader):
        
            ### PREPARE MINIBATCH
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
                
            ### FORWARD AND BACK PROP
            logits = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 300:
                if printing:
                    print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                        f'Batch {batch_idx:03d}/{len(train_loader):03d} |' 
                        f' Cost: {cost:.4f}')

        # no need to build the computation graph for backprop when computing accuracy
        model.eval()
        with torch.set_grad_enabled(False):
            train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
            valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
            train_acc_lst.append(train_acc)
            valid_acc_lst.append(valid_acc)
            train_loss_lst.append(train_loss)
            valid_loss_lst.append(valid_loss)
            if printing:
                print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
                    f' | Validation Acc.: {valid_acc:.2f}%')
            
        elapsed = (time.time() - start_time)/60
        if printing:
            print(f'Time elapsed: {elapsed:.2f} min')

    elapsed = (time.time() - start_time)/60
    if saving:
        torch.save(model.state_dict(), PATH)
    if printing:
        print(f'Total Training Time: {elapsed:.2f} min')


###################################################################################

def main(modelname,num,p,num_classes,freezeornot,epo):
    
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
    
    torch.save(model.state_dict(), PATH_new)
    model.load_state_dict(models7a.get_weight(PATH_random=PATH_new,PATH_standard=PATH_PARA,partition=p,whichnn=modelname))

    NUM_EPOCHS = epo
    #DEVICE = torch.device("cpu")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH="../data2/"+modelname+"-"+str(p)+"-"+str(freezeornot)+".pkl"

    train_loader,test_loader = load_data(Size=num)

    train(model,DEVICE,NUM_EPOCHS,train_loader,PATH,test_loader,printing=False,saving=True)
    print(num,p,modelname)

def standard_training(num,num_classes,modelname):
    if   modelname=="vgg19":
        model = models7a.vgg19(num_classes=num_classes)
        NUM_EPOCHS = 40
    elif modelname=="vgg13":
        model = models7a.vgg13(num_classes=num_classes)
        NUM_EPOCHS = 35

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH="../data2/standard_"+modelname+"-"+str(num_classes)+"-"+str(num)+".pkl"

    print(num)
    train_loader,test_loader = load_data(Size=num)
    train(model,DEVICE,NUM_EPOCHS,train_loader,PATH,test_loader,printing=False,saving=True)


suggest_train_size=[3000,2500,4000]
suggest_epo_up=[[31,22,17,7,2,4],
[76,36,15,7,10,17],
[28,21,11,8,8,20]]

suggest_epo=[[24,22,15,5,2,4],
[56,36,15,7,5,5],
[22,16,10,6,4,3]]

suggest_epo_down=[[21,18,10,2,2,2],
[52,28,12,5,4,3],
[22,11,10,4,4,3]]

partitions=[0,20,40,60,80,100]
models    =["ennclave","soter","aegisdnn"]
if __name__=='__main__':  
    time1=time.time()
    standard_training(num=2800,num_classes=100,modelname="vgg13")
    standard_training(num=18000,num_classes=100,modelname="vgg19")
    standard_training(num=3000,num_classes=10,modelname="vgg13")
    standard_training(num=21000,num_classes=10,modelname="vgg19")

    for i in range(6):
        main("ennclave",suggest_train_size[0],partitions[i],100,False,suggest_epo[0][i])

    for i in range(6):
        main("soter",suggest_train_size[1],partitions[i],100,False,suggest_epo[1][i])

    for i in range(6):
        main("aegisdnn",suggest_train_size[2],partitions[i],100,False,suggest_epo[2][i])

    time2=time.time()-time1
    print("total time:",time2/60)


################################################## ends #####################################