import torch
import torchvision
import torch.nn as nn  
import copy
import random

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,Dataset,size):
        self.data = Dataset
        self.len=min(self.data.__len__(),size)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return self.len


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

'''odict_keys(['feature1.0.weight', 'feature1.0.bias', 'feature2.0.weight', 'feature2.0.bias', 'feature2.0.running_mean', 'feature2.0.running_var', 'feature2.0.num_batches_tracked', 'feature3.0.weight', 'feature3.0.bias', 'feature4.0.weight', 'feature4.0.bias', 'feature4.0.running_mean', 'feature4.0.running_var', 'feature4.0.num_batches_tracked', 'feature5.0.weight', 'feature5.0.bias', 'feature6.0.weight', 'feature6.0.bias', 'feature6.0.running_mean', 'feature6.0.running_var', 'feature6.0.num_batches_tracked', 'feature7.0.weight', 'feature7.0.bias', 'feature8.0.weight', 'feature8.0.bias', 'feature8.0.running_mean', 'feature8.0.running_var', 'feature8.0.num_batches_tracked', 'feature9.0.weight', 'feature9.0.bias', 'feature10.0.weight', 'feature10.0.bias', 'feature10.0.running_mean', 'feature10.0.running_var', 'feature10.0.num_batches_tracked', 'feature11.0.weight', 'feature11.0.bias', 'feature12.0.weight', 'feature12.0.bias', 'feature12.0.running_mean', 'feature12.0.running_var', 'feature12.0.num_batches_tracked', 'feature13.0.weight', 'feature13.0.bias', 'feature14.0.weight', 'feature14.0.bias', 'feature14.0.running_mean', 'feature14.0.running_var', 'feature14.0.num_batches_tracked', 'feature15.0.weight', 'feature15.0.bias', 'feature16.0.weight', 'feature16.0.bias', 'feature16.0.running_mean', 'feature16.0.running_var', 'feature16.0.num_batches_tracked', 'feature17.0.weight', 'feature17.0.bias', 'feature18.0.weight', 'feature18.0.bias', 'feature18.0.running_mean', 'feature18.0.running_var', 'feature18.0.num_batches_tracked', 'feature19.0.weight', 'feature19.0.bias', 'feature20.0.weight', 'feature20.0.bias', 'feature20.0.running_mean', 'feature20.0.running_var', 'feature20.0.num_batches_tracked', 'feature21.0.weight', 'feature21.0.bias', 'feature22.0.weight', 'feature22.0.bias', 'feature22.0.running_mean', 'feature22.0.running_var', 'feature22.0.num_batches_tracked', 'feature23.0.weight', 'feature23.0.bias', 'feature24.0.weight', 'feature24.0.bias', 'feature24.0.running_mean', 'feature24.0.running_var', 'feature24.0.num_batches_tracked', 'feature25.0.weight', 'feature25.0.bias', 'feature26.0.weight', 'feature26.0.bias', 'feature26.0.running_mean', 'feature26.0.running_var', 'feature26.0.num_batches_tracked', 'feature27.0.weight', 'feature27.0.bias', 'feature28.0.weight', 'feature28.0.bias', 'feature28.0.running_mean', 'feature28.0.running_var', 'feature28.0.num_batches_tracked', 'feature29.0.weight', 'feature29.0.bias', 'feature30.0.weight', 'feature30.0.bias', 'feature30.0.running_mean', 'feature30.0.running_var', 'feature30.0.num_batches_tracked', 'feature31.0.weight', 'feature31.0.bias', 'feature32.0.weight', 'feature32.0.bias', 'feature32.0.running_mean', 'feature32.0.running_var', 'feature32.0.num_batches_tracked', 'classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias'])'''
'''odict_keys(['feature1.0.weight', 'feature1.0.bias', 'feature2.0.weight', 'feature2.0.bias', 'feature2.0.running_mean', 'feature2.0.running_var', 'feature2.0.num_batches_tracked', 'feature3.0.weight', 'feature3.0.bias', 'feature4.0.weight', 'feature4.0.bias', 'feature4.0.running_mean', 'feature4.0.running_var', 'feature4.0.num_batches_tracked', 'feature5.0.weight', 'feature5.0.bias', 'feature6.0.weight', 'feature6.0.bias', 'feature6.0.running_mean', 'feature6.0.running_var', 'feature6.0.num_batches_tracked', 'feature7.0.weight', 'feature7.0.bias', 'feature8.0.weight', 'feature8.0.bias', 'feature8.0.running_mean', 'feature8.0.running_var', 'feature8.0.num_batches_tracked', 'feature9.0.weight', 'feature9.0.bias', 'feature10.0.weight', 'feature10.0.bias', 'feature10.0.running_mean', 'feature10.0.running_var', 'feature10.0.num_batches_tracked', 'feature11.0.weight', 'feature11.0.bias', 'feature12.0.weight', 'feature12.0.bias', 'feature12.0.running_mean', 'feature12.0.running_var', 'feature12.0.num_batches_tracked', 'feature13.0.weight', 'feature13.0.bias', 'feature14.0.weight', 'feature14.0.bias', 'feature14.0.running_mean', 'feature14.0.running_var', 'feature14.0.num_batches_tracked', 'feature15.0.weight', 'feature15.0.bias', 'feature16.0.weight', 'feature16.0.bias', 'feature16.0.running_mean', 'feature16.0.running_var', 'feature16.0.num_batches_tracked', 'feature17.0.weight', 'feature17.0.bias', 'feature18.0.weight', 'feature18.0.bias', 'feature18.0.running_mean', 'feature18.0.running_var', 'feature18.0.num_batches_tracked', 'feature19.0.weight', 'feature19.0.bias', 'feature20.0.weight', 'feature20.0.bias', 'feature20.0.running_mean', 'feature20.0.running_var', 'feature20.0.num_batches_tracked', 'classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias'])'''

class vgg19(nn.Module):
    def __init__(self, num_classes=1000):
        super(vgg19, self).__init__()
        self.feature1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1))
        self.feature2 = nn.Sequential(nn.BatchNorm2d(num_features=64),              nn.ReLU())
        self.feature3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature4 = nn.Sequential(nn.BatchNorm2d(num_features=64),              nn.ReLU(),            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.feature5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature6 = nn.Sequential(nn.BatchNorm2d(num_features=128),             nn.ReLU(),)
        self.feature7 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature8 = nn.Sequential(nn.BatchNorm2d(num_features=128),             nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.feature9 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature10 = nn.Sequential(nn.BatchNorm2d(num_features=256),            nn.ReLU())
        self.feature11 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature12 = nn.Sequential(nn.BatchNorm2d(num_features=256),            nn.ReLU())
        self.feature13 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature14 = nn.Sequential(nn.BatchNorm2d(num_features=256),            nn.ReLU())
        self.feature15 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature16 = nn.Sequential(nn.BatchNorm2d(num_features=256),            nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.feature17 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature18 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature19 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature20 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature21 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature22 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature23 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature24 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.feature25 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature26 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature27 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature28 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature29 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature30 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature31 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature32 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.feature5(x)
        x = self.feature6(x)
        x = self.feature7(x)
        x = self.feature8(x)
        x = self.feature9(x)
        x = self.feature10(x)
        x = self.feature11(x)
        x = self.feature12(x)
        x = self.feature13(x)
        x = self.feature14(x)
        x = self.feature15(x)
        x = self.feature16(x)
        x = self.feature17(x)
        x = self.feature18(x)
        x = self.feature19(x)
        x = self.feature20(x)
        x = self.feature21(x)
        x = self.feature22(x)
        x = self.feature23(x)
        x = self.feature24(x)
        x = self.feature25(x)
        x = self.feature26(x)
        x = self.feature27(x)
        x = self.feature28(x)
        x = self.feature29(x)
        x = self.feature30(x)
        x = self.feature31(x)
        x = self.feature32(x)

        outputs = self.classifier(x.view(-1,512*1*1))
        return outputs

class vgg13(nn.Module):
    def __init__(self, num_classes=1000):
        super(vgg13, self).__init__()
        self.feature1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1))
        self.feature2 = nn.Sequential(nn.BatchNorm2d(num_features=64),              nn.ReLU())
        self.feature3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature4 = nn.Sequential(nn.BatchNorm2d(num_features=64),              nn.ReLU(),            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.feature5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature6 = nn.Sequential(nn.BatchNorm2d(num_features=128),             nn.ReLU(),)
        self.feature7 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature8 = nn.Sequential(nn.BatchNorm2d(num_features=128),             nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.feature9 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature10 = nn.Sequential(nn.BatchNorm2d(num_features=256),            nn.ReLU())
        self.feature11 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature12 = nn.Sequential(nn.BatchNorm2d(num_features=256),            nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.feature13 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature14 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature15 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature16 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.feature17 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature18 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature19 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature20 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.feature5(x)
        x = self.feature6(x)
        x = self.feature7(x)
        x = self.feature8(x)
        x = self.feature9(x)
        x = self.feature10(x)
        x = self.feature11(x)
        x = self.feature12(x)
        x = self.feature13(x)
        x = self.feature14(x)
        x = self.feature15(x)
        x = self.feature16(x)
        x = self.feature17(x)        
        x = self.feature18(x)        
        x = self.feature19(x)
        x = self.feature20(x)
        
        outputs = self.classifier(x.view(-1,512*1*1))
        return outputs

weightname13=['feature1.0.weight', 'feature1.0.bias', 'feature3.0.weight', 'feature3.0.bias',
'feature5.0.weight', 'feature5.0.bias', 'feature7.0.weight', 'feature7.0.bias', 'feature9.0.weight', 'feature9.0.bias',
'feature11.0.weight', 'feature11.0.bias',   'feature13.0.weight', 'feature13.0.bias', 
'feature15.0.weight', 'feature15.0.bias',  'feature17.0.weight', 'feature17.0.bias', 
'feature19.0.weight', 'feature19.0.bias',   'classifier.0.weight', 'classifier.0.bias',
 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']

weightname19=['feature1.0.weight', 'feature1.0.bias', 'feature3.0.weight', 'feature3.0.bias',
'feature5.0.weight', 'feature5.0.bias', 'feature7.0.weight', 'feature7.0.bias', 'feature9.0.weight', 'feature9.0.bias',
'feature11.0.weight', 'feature11.0.bias',   'feature17.0.weight', 'feature17.0.bias', 
'feature19.0.weight', 'feature19.0.bias',  'feature29.0.weight', 'feature29.0.bias', 
'feature31.0.weight', 'feature31.0.bias',   'classifier.0.weight', 'classifier.0.bias',
'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']


def get_weight(PATH_random, PATH_standard, partition, whichnn, morph_n=3):
    weights = torch.load(PATH_random)
    weight0 = torch.load(PATH_standard)

    n=1
    if whichnn == "soter":
        n=morph_n
    if partition >= 20:
        weights['feature1.0.weight']=weight0['feature1.0.weight']*n
        weights['feature1.0.bias']=weight0['feature1.0.bias']*n
        weights['feature3.0.weight']=weight0['feature3.0.weight']*n
        weights['feature3.0.bias']=weight0['feature3.0.bias']*n
        if partition >= 40:
            weights['feature5.0.weight']=weight0['feature5.0.weight']*n
            weights['feature5.0.bias']=weight0['feature5.0.bias']*n
            weights['feature7.0.weight']=weight0['feature7.0.weight']*n
            weights['feature7.0.bias']=weight0['feature7.0.bias']*n
            if partition >= 60:
                weights['feature9.0.weight']=weight0['feature9.0.weight']*n
                weights['feature9.0.bias']=weight0['feature9.0.bias']*n
                weights['feature11.0.weight']=weight0['feature11.0.weight']*n
                weights['feature11.0.bias']=weight0['feature11.0.bias']*n
                if whichnn == "ennclave":
                    weights['feature13.0.weight']=weight0['feature13.0.weight']
                    weights['feature13.0.bias']=weight0['feature13.0.bias']
                elif whichnn == "soter" or whichnn == "aegisdnn":
                    weights['feature13.0.weight']=weight0['feature17.0.weight']
                    weights['feature13.0.bias']=weight0['feature17.0.bias']
                if partition >= 80:
                    if whichnn == "ennclave":
                        weights['feature15.0.weight']=weight0['feature15.0.weight']
                        weights['feature15.0.bias']=weight0['feature15.0.bias']
                        weights['feature17.0.weight']=weight0['feature17.0.weight']
                        weights['feature17.0.bias']=weight0['feature17.0.bias']
                        weights['feature19.0.weight']=weight0['feature19.0.weight']
                        weights['feature19.0.bias']=weight0['feature19.0.bias']
                    elif whichnn == "soter" or whichnn == "aegisdnn":
                        weights['feature15.0.weight']=weight0['feature19.0.weight']*n
                        weights['feature15.0.bias']=weight0['feature19.0.bias']*n
                        weights['feature17.0.weight']=weight0['feature29.0.weight']*n
                        weights['feature17.0.bias']=weight0['feature29.0.bias']*n
                        weights['feature19.0.weight']=weight0['feature31.0.weight']*n
                        weights['feature19.0.bias']=weight0['feature31.0.bias']*n
                    if partition >= 100:
                        if whichnn == "ennclave":
                            weights['classifier.0.weight']=weight0['classifier.0.weight']
                            weights['classifier.0.bias']=weight0['classifier.0.bias']
                            weights['classifier.3.weight']=weight0['classifier.3.weight']
                            weights['classifier.3.bias']=weight0['classifier.3.bias']
                            weights['classifier.6.weight']=weight0['classifier.6.weight']
                            weights['classifier.6.bias']=weight0['classifier.6.bias']
                        if whichnn == "ennclave":
                            weights['classifier.0.weight']=weight0['classifier.0.weight']*n
                            weights['classifier.0.bias']=weight0['classifier.0.bias']*n
                            weights['classifier.3.weight']=weight0['classifier.3.weight']*n
                            weights['classifier.3.bias']=weight0['classifier.3.bias']*n
                            weights['classifier.6.weight']=weight0['classifier.6.weight']*n
                            weights['classifier.6.bias']=weight0['classifier.6.bias']*n
    return weights

def get_weight_random(PATH_random, PATH_standard, partition, whichnn, morph_n=3):
    
    weights = torch.load(PATH_random)
    weight0 = torch.load(PATH_standard)

    n=1
    if whichnn == "soter":
        n=morph_n
    
    lists=[]

    if whichnn == "ennclave":
        for i in range(int(partition/100*12)):
            rep = random.randint(0,12)
            while rep in lists:
                rep = random.randint(0,12)
            weights[weightname13[2*rep]]=weight0[weightname13[rep*2]]
            weights[weightname13[2*rep+1]]=weight0[weightname13[rep*2+1]]
            lists.append(rep)
    else:
        for i in range(int(partition/100*12)):
            rep = random.randint(0,12)
            while rep in lists:
                rep = random.randint(0,12)
            weights[weightname13[2*rep]]=weight0[weightname19[rep*2]]*n
            weights[weightname13[2*rep+1]]=weight0[weightname19[rep*2+1]]*n
            lists.append(rep)
    

    return weights

class vgg13_freeze(nn.Module):
    def __init__(self, num_classes=1000,partition=0):
        super(vgg13_freeze, self).__init__()
        self.feature1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1))
        self.feature2 = nn.Sequential(nn.BatchNorm2d(num_features=64),              nn.ReLU())
        self.feature3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature4 = nn.Sequential(nn.BatchNorm2d(num_features=64),              nn.ReLU(),            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.feature5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature6 = nn.Sequential(nn.BatchNorm2d(num_features=128),             nn.ReLU(),)
        self.feature7 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature8 = nn.Sequential(nn.BatchNorm2d(num_features=128),             nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.feature9 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature10 = nn.Sequential(nn.BatchNorm2d(num_features=256),            nn.ReLU())
        self.feature11 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature12 = nn.Sequential(nn.BatchNorm2d(num_features=256),            nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.feature13 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature14 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature15 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature16 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.feature17 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature18 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU())
        self.feature19 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.feature20 = nn.Sequential(nn.BatchNorm2d(num_features=512),            nn.ReLU(),            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,num_classes)
        )
        if partition >= 20:
            freeze(self.feature1)
            freeze(self.feature3)
            if partition >= 40:
                freeze(self.feature5)
                freeze(self.feature7)
                if partition >= 60:
                    freeze(self.feature9)
                    freeze(self.feature11)
                    #freeze(self.feature13)
                    if partition >= 80:
                        freeze(self.feature13)
                        freeze(self.feature15)
                        #freeze(self.feature17)
                        #freeze(self.feature19)
                        if partition >= 100:
                            freeze(self.feature17)
                            freeze(self.feature19)
                            #freeze(self.classifier)

    def forward(self,x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.feature5(x)
        x = self.feature6(x)
        x = self.feature7(x)
        x = self.feature8(x)
        x = self.feature9(x)
        x = self.feature10(x)
        x = self.feature11(x)
        x = self.feature12(x)
        x = self.feature13(x)
        x = self.feature14(x)
        x = self.feature15(x)
        x = self.feature16(x)
        x = self.feature17(x)        
        x = self.feature18(x)        
        x = self.feature19(x)
        x = self.feature20(x)
        
        outputs = self.classifier(x.view(-1,512*1*1))
        return outputs




if __name__=='__main__':
    print("Please include this file;")
    print("MyDataset is to select certain data from CIFAR10/CIFAR100")
    print("vgg13 and vgg19 is standard vgg")
    print("vgg13_freeze can freeze certain layers in a static way")
    print("get_weight() and get_weight_random() helps repleace parameters")
    print("--------------------------------------------------------------")
