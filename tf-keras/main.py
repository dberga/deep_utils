# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import os
CUDA_VERSION=10.0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #use nvidia-smi bus id order
os.environ["CUDA_VISIBLE_DEVICES"]=str(1) #using nvidia ID 1
os.environ['PATH'] += ':/usr/local/cuda%s/bin/'%str(CUDA_VERSION); #path to cuda10.0
os.environ['LD_LIBRARY_PATH']=  ':/usr/local/cuda%s/lib64/'%str(CUDA_VERSION) + ':/usr/lib/x86_64-linux-gnu/'; #path to cuda/cudnn lib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #remove useless logs


from ipywidgets import IntProgress #necessary for some dataset downloader
import pickle as pkl #to save/load files

import plot_utils
import preprocess_utils
import model_utils
import learning_utils
import pdb #debug

################################################## HYPERPARAMETERS and CONSTANTS ##################################################

import sys
if(len(sys.argv)==1):
    sys.argv=[sys.argv[0],'resnet18']
SELECTED_MODEL=str(sys.argv[1])
VISUALIZE=False
SHUFFLE=False

BATCH_SIZE=6 #128
EPOCHS=200
base_LR=0.001
fin_LR=0.01 #multiLR (0.001 to 0.01)
OPTIMIZER_TYPE='SGD' #SGDmult
LOSS='sparse_categorical_crossentropy' #original: 'categorical_crossentropy'
METRICS=['accuracy']
PRETRAINING=None #'imagenet' #'imagenet'

################################################## DOWNLOAD AND PREPARE DATASET ##################################################
## Fetch the dataset by dataset string name or by call to class function
tfds.list_builders()
builder = tfds.builder('oxford_flowers102') #cifar100, caltech_birds2011, oxford_flowers102
#builder = tfds.image.oxford_flowers102.OxfordFlowers102()


## Describe the dataset with DatasetInfo
#print(builder.info.features['image'].shape)
num_classes=builder.info.features['label'].num_classes
num_training_samples=builder.info.splits['train'].num_examples
num_test_samples=builder.info.splits['test'].num_examples
#num_val_samples=builder.info.splits['validation'].num_examples
label_names=builder.info.features['label'].names
print("Num classes:"); print(num_classes)
print("Labels:"); print(label_names)
print("Train samples: "); print(num_training_samples)
#print("Validation samples: "); print(num_val_samples)
print("Test samples: "); print(num_test_samples)

## Download the data, prepare it, and write it to disk
builder.download_and_prepare()

################################################## PREPROCESS DATASET AND SPLIT DATA ##################################################

## Load data from disk as tf.data.Datasets
dataset = builder.as_dataset(
      batch_size=-1,  #use batch_size on fit function
      shuffle_files=SHUFFLE,
      decoders={'image': preprocess_utils.decode_image()},
      as_supervised=True) #split=['train', 'test'], in_memory=True

#split data
train_dataset=dataset['train']
try:
    val_dataset=dataset['validation']
except:
    #pdb.set_trace()
    train_dataset=dataset['train[:80%]']
    val_dataset=dataset['train[-20%:]']
test_dataset=dataset['test']
np_train_dataset = tfds.as_numpy(train_dataset) 
np_test_dataset = tfds.as_numpy(test_dataset)
np_val_dataset = tfds.as_numpy(val_dataset)
x_train,y_train=np_train_dataset[0],np_train_dataset[1]
x_val,y_val=np_val_dataset[0],np_val_dataset[1]
x_test,y_test=np_test_dataset[0],np_test_dataset[1]

#create alternative b/w samples
dataset_bw = builder.as_dataset(
      batch_size=-1,  #use batch_size on fit function
      shuffle_files=True,
      decoders={'image': preprocess_utils.decode_image_bw()},
      as_supervised=True) #split=['train', 'test'], in_memory=True

#split data
train_dataset_bw=dataset_bw['train']
try:
    val_dataset_bw=dataset_bw['validation']
except:
    train_dataset_bw=dataset_bw['train[:80%]']
    val_dataset_bw=dataset_bw['train[-20%:]']
test_dataset_bw=dataset_bw['test']
np_train_dataset_bw = tfds.as_numpy(train_dataset_bw) 
np_test_dataset_bw = tfds.as_numpy(test_dataset_bw)
np_val_dataset_bw = tfds.as_numpy(val_dataset_bw)
x_train_bw,y_train_bw=np_train_dataset_bw[0],np_train_dataset_bw[1]
x_val_bw,y_val_bw=np_val_dataset_bw[0],np_val_dataset_bw[1]
x_test_bw,y_test_bw=np_test_dataset_bw[0],np_test_dataset_bw[1]


######################### Visualize each batch #########################

#fig = tfds.show_examples(builder.info, train_dataset)

if VISUALIZE==True:
    #if BATCH_SIZE == -1:
    #    plot_utils.show_batch(x_train,y_train,builder.info.features['label'].names)
    #else:
    for batch in np_train_dataset:
        images, labels = batch[0], batch[1]
        #print(np.shape(images))
        #print(labels)
        plot_utils.show_batch(images,labels,builder.info.features['label'].names)



################################################## Create/Compile model graph ##################################################

MODELS_LIST=['resnet18_bw','resnet18','resnet18_combined']  #'bw','rgb','rgb_3by3','combined',,'resnet18_rgb','ResNet18 RGB(1x1)'
LEGEND=['ResNet18 B/W(rinit)','ResNet18(rinit)','ResNet18 B/W+RGB(rinit)']  #'B/W(3x3)','RGB(1x1)','RGB(3x3)','B/W(3x3)+RGB(1x1)',

#plot existing results
if SELECTED_MODEL=='plot':
    plot_utils.plot_all_history(MODELS_LIST,LEGEND)
    exit()

#########################prepare training data
if SELECTED_MODEL=='rgb' or SELECTED_MODEL=='rgb_3by3' or SELECTED_MODEL=='resnet18' or SELECTED_MODEL=='resnet18_rgb': #single branch, color X
    X,Y=x_train,y_train
    validation_data=((x_val,y_val))
    test_data=[x_test,y_test]
    #callbacks=plot_utils.TestCallback(x_test)
elif SELECTED_MODEL=='bw' or SELECTED_MODEL=='resnet18_bw': #single branch, grayscale X
    X,Y=x_train_bw,y_train
    validation_data=((x_val_bw,y_val))
    test_data=[x_test_bw,y_test]
elif SELECTED_MODEL=='combined' or SELECTED_MODEL=='resnet18_combined': #two branch, color X
    X,Y=[x_train,x_train_bw],y_train
    validation_data=(([x_val,x_val_bw],y_val))
    test_data=[[x_test,x_test_bw],y_test]
    #callbacks=plot_utils.TestCallback(x_test_bw)
    
######################### get model tensors
    
if SELECTED_MODEL=='bw':
    model=model_utils.KModel_3x3conv(INPUT_SHAPE=(preprocess_utils.RZ_IMG_HEIGHT,preprocess_utils.RZ_IMG_WIDTH,3),OUTPUT_SHAPE=(num_classes))
elif SELECTED_MODEL=='rgb':
    model=model_utils.KModel_1x1conv(INPUT_SHAPE=(preprocess_utils.RZ_IMG_HEIGHT,preprocess_utils.RZ_IMG_WIDTH,3),OUTPUT_SHAPE=(num_classes))
elif SELECTED_MODEL=='rgb_3by3':
    model=model_utils.KModel_3x3conv(INPUT_SHAPE=(preprocess_utils.RZ_IMG_HEIGHT,preprocess_utils.RZ_IMG_WIDTH,3),OUTPUT_SHAPE=(num_classes))
elif SELECTED_MODEL=='combined':
    model=model_utils.KModel_1x1combi3x3(INPUT_SHAPE=(preprocess_utils.RZ_IMG_HEIGHT,preprocess_utils.RZ_IMG_WIDTH,3),OUTPUT_SHAPE=(num_classes))
elif SELECTED_MODEL=='resnet18_bw':
    model=model_utils.KModel_resnet18(INPUT_SHAPE=(preprocess_utils.RZ_IMG_HEIGHT,preprocess_utils.RZ_IMG_WIDTH,3),OUTPUT_SHAPE=(num_classes),pretrained=PRETRAINING,top_layers=False)
elif SELECTED_MODEL=='resnet18_rgb':
    model=model_utils.KModel_resnet18_1x1conv(INPUT_SHAPE=(preprocess_utils.RZ_IMG_HEIGHT,preprocess_utils.RZ_IMG_WIDTH,3),OUTPUT_SHAPE=(num_classes),pretrained=PRETRAINING,top_layers=False)
elif SELECTED_MODEL=='resnet18':    
    model=model_utils.KModel_resnet18(INPUT_SHAPE=(preprocess_utils.RZ_IMG_HEIGHT,preprocess_utils.RZ_IMG_WIDTH,3),OUTPUT_SHAPE=(num_classes),pretrained=PRETRAINING,top_layers=False)
elif SELECTED_MODEL=='resnet18_combined':    
    model=model_utils.KModel_resnet18_1x1combi3x3(INPUT_SHAPE=(preprocess_utils.RZ_IMG_HEIGHT,preprocess_utils.RZ_IMG_WIDTH,3),OUTPUT_SHAPE=(num_classes),pretrained=PRETRAINING,top_layers=False)

######################### select optimizer, loss and compile model

multipliers=learning_utils.getMultipliers(model,base_LR,fin_LR)

OPTIMIZER=learning_utils.SelectOptimizer(opt=OPTIMIZER_TYPE,LR=base_LR,multipliers=multipliers)
model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=METRICS
          ) 

#test callback
#callbacks=[plot_utils.TestCallback(test_data)]
callbacks=None

#########################print model structure and plot model
model.summary()
#pdb.set_trace()
tf.keras.utils.plot_model(model,to_file="model_%s.png"%(SELECTED_MODEL))

################################################## Train & test model ##################################################

#train

history=model.fit(x=X,y=Y,validation_data=validation_data,batch_size=BATCH_SIZE,callbacks=callbacks, epochs=EPOCHS,verbose=2)

#test
test_loss, test_acc = model.evaluate(x=test_data[0],  y=test_data[1], batch_size=BATCH_SIZE, verbose=2)
print("Test loss: %s, Test accuracy: %s"%(str(test_loss),str(test_acc)))

with open("history_%s.pkl"%SELECTED_MODEL,'wb') as f:
        pkl.dump(history.history, f)

        
######################### Plot history (accuracy and loss) #########################


plot_utils.plot_all_history(MODELS_LIST,LEGEND)



