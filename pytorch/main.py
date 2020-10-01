

#wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
#tar -xvf flower_data.tar.gz flowers

#code adapted from https://github.com/Muhammad-MujtabaSaeed/102-Flowers-Classification/blob/master/102_Flowers_classification.ipynb


import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
import json
import urllib.request
import copy
import seaborn as sns
from PIL import Image
from collections import OrderedDict
#matplotlib inline
#config InlineBackend.figure_format = 'retina'

import os
CUDA_VERSION=10.0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #use nvidia-smi bus id order
os.environ["CUDA_VISIBLE_DEVICES"]=str(1) #using nvidia ID 1
os.environ['PATH'] += ':/usr/local/cuda%s/bin/'%str(CUDA_VERSION); #path to cuda10.0
os.environ['LD_LIBRARY_PATH']=  ':/usr/local/cuda%s/lib64/'%str(CUDA_VERSION) + ':/usr/lib/x86_64-linux-gnu/'; #path to cuda/cudnn lib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #remove useless logs

import visualize_utils
import training_utils


################################################## PREPARE DATASET ##################################################


data_dir = './flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])
    }

image_datasets = {
        x: datasets.ImageFolder(root=data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }

dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
        for x in list(image_datasets.keys())
    }
dataset_sizes = {
        x: len(dataloaders[x].dataset) 
        for x in list(image_datasets.keys())
    } 
class_names = image_datasets['train'].classes

dataset_sizes

urllib.request.urlretrieve('https://raw.githubusercontent.com/Muhammad-MujtabaSaeed/102-Flowers-Classification/master/cat_to_name.json','cat_to_name.json')

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
for i in range(0,len(class_names)):
    class_names[i] = cat_to_name.get(class_names[i])
    

inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)



################################################## VISUALIZE DATASET ##################################################


#visualize_utils.imshow(out, title=[class_names[x] for x in classes])


################################################## PREPARE MODEL ##################################################


model_ft = models.resnet18(pretrained=True) # loading a pre-trained(trained on image net) resnet18 model from torchvision models
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 102) # changing the last layer for this dataset by setting last layer neurons to 102 as this dataset has 102 categories

################################################## PREPARE OPTIMIZER ##################################################


try:
    checkpoint = torch.load('point_resnet_best.pth')
    model_ft.load_state_dict(checkpoint['model'])
    optimizer_ft.load_state_dict(checkpoint['optim'])
except:
    criterion = nn.CrossEntropyLoss() # defining loss function
    use_gpu = torch.cuda.is_available() # if gpu is available then use it
    if use_gpu:
        model_ft = model_ft.cuda()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9) # defining optimizer with learning rate set to 0.0001


################################################## TRAIN ##################################################


#model_ft = training_utils.train_model(model_ft, criterion, optimizer_ft,dataloaders=dataloaders,use_gpu=use_gpu,dataset_sizes=dataset_sizes,num_epochs=20)

################################################## VISUALIZE ##################################################

#visualize_utils.visualize_model(model=model_ft,dataloaders=dataloaders,use_gpu=use_gpu,class_names=class_names,num_images=8)


################################################## VISUALIZE ##################################################



top1 ,top5 = training_utils.calc_accuracy(model_ft, 'test',dataloaders,use_gpu,dataset_sizes)
print(float(top1.avg))
print(float(top5.avg))


