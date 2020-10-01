# -*- coding: utf-8 -*-

#from tensorflow.keras
#from keras

from keras import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
import pdb

from keras.models import model_from_json
        
def KModel_1x1conv(INPUT_SHAPE=(224, 224, 3),OUTPUT_SHAPE=(10)):
    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1,1), padding='same',activation='relu',input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu'))
    model.add(Flatten())
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(OUTPUT_SHAPE, activation='softmax'))
    return model

def KModel_3x3conv(INPUT_SHAPE=(224, 224, 3),OUTPUT_SHAPE=(10)):
    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu',input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
    model.add(Flatten())
    #model.add(Dense(64, activation='relu'))
    
    model.add(Dense(OUTPUT_SHAPE, activation='softmax'))
    return model



def KModel_1x1combi3x3(INPUT_SHAPE=(224, 224, 3),OUTPUT_SHAPE=(10)):
    
    INPUT_SHAPE_RGB=(INPUT_SHAPE[0],INPUT_SHAPE[1],3)
    INPUT_SHAPE_BW=(INPUT_SHAPE[0],INPUT_SHAPE[1],1)
    rgbinput=Input(shape=INPUT_SHAPE_RGB)
    bwinput=Input(shape=INPUT_SHAPE_BW)
    
    rgbmodel=rgbinput
    rgbmodel=Conv2D(filters=32, kernel_size=(1, 1), strides=(1,1), padding='same',activation='relu')(rgbmodel)
    rgbmodel=Conv2D(filters=32, kernel_size=(1, 1), strides=(1,1), activation='relu')(rgbmodel)
    rgbmodel=MaxPooling2D(pool_size=(2, 2))(rgbmodel)
    rgbmodel=Conv2D(filters=32, kernel_size=(1, 1), strides=(1,1), padding='same',activation='relu')(rgbmodel)
    rgbmodel=Flatten()(rgbmodel)
    #rgbmodel=Dense(64, activation='relu')(rgbmodel)
    
    
    bwmodel=bwinput
    bwmodel=Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1),padding='same',activation='relu')(bwmodel)
    bwmodel=Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), activation='relu')(bwmodel)
    bwmodel=MaxPooling2D(pool_size=(2, 2))(bwmodel)
    bwmodel=Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')(bwmodel)
    bwmodel=Flatten()(bwmodel)
    #bwmodel=Dense(64, activation='relu')(bwmodel)
    combined=Concatenate()([rgbmodel, bwmodel])
    combined=Dense(OUTPUT_SHAPE, activation="softmax")(combined)
    
    
    model=Model(inputs=[rgbinput,bwinput],outputs=combined)
    
    return model

#comment these if using bw, rgb, rgb_3by3 and combined
from classification_models.keras import Classifiers
#from keras.layers import Dense
#from keras import Model

def KModel_resnet18(INPUT_SHAPE=(224, 224, 3),OUTPUT_SHAPE=(10), pretrained='imagenet', top_layers=True):
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    model=ResNet18(input_shape=INPUT_SHAPE, classes=OUTPUT_SHAPE, weights=pretrained,include_top=top_layers)
        
    if top_layers is False:
        out_gaPool = GlobalAveragePooling2D()(model.output)
        out_softmax = Dense(OUTPUT_SHAPE,activation='softmax')(out_gaPool)
        #out_softmax = Activation('softmax')(out_gaPool)
        model = Model(inputs=[model.input], outputs=[out_softmax])

    #model=change_strides(model,target_strides=(1,1))
    model=change_padding(model,padding='same')
    return model

def KModel_resnet18_1x1conv(INPUT_SHAPE=(224, 224, 3),OUTPUT_SHAPE=(10), pretrained='imagenet', top_layers=True):
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    model=ResNet18(input_shape=INPUT_SHAPE, classes=OUTPUT_SHAPE, weights=None,include_top=top_layers)
    
    if top_layers is False:
        out_gaPool = GlobalAveragePooling2D()(model.output)
        out_softmax = Dense(OUTPUT_SHAPE,activation='softmax')(out_gaPool)
        #out_softmax = Activation('softmax')(out_gaPool)
        model = Model(inputs=[model.input], outputs=[out_softmax])
    model=change_kernelsizes(model,target_kernel_size=(1,1))
    #model=change_strides(model,target_strides=(1,1))
    model=change_padding(model,padding='same')
    return model
    
def change_kernelsizes(model,target_kernel_size=(1,1)):
    for i in range(len(model.layers)):
        if "conv" in model.layers[i].name.lower():
            #conf=model.layers[i].get_config()
            #conf['kernel_size']=target_kernel_size
            #model.layers[i].from_config(conf)
            model.layers[i].kernel_size=target_kernel_size
    #model_json = model.to_json()
    #model=model_from_json(model_json)
    return model

def change_strides(model,target_strides=(1,1)):
    for i in range(len(model.layers)):
        if "conv" in model.layers[i].name.lower():
            #conf=model.layers[i].get_config()
            #conf['strides']=target_strides
            #model.layers[i].from_config(conf)
            model.layers[i].strides=target_strides
    #model_json = model.to_json()
    #model=model_from_json(model_json)
    return model
def change_padding(model,padding='same'):
    for i in range(len(model.layers)):
        if "conv" in model.layers[i].name.lower():
            #conf=model.layers[i].get_config()
            #conf['padding']='same'
            #model.layers[i].from_config(conf)
            model.layers[i].padding=padding
    #model_json = model.to_json()
    #model=model_from_json(model_json)
    return model

def add_layers_name_suffix(model,suffix='rgb'):
    for i in range(len(model.layers)):
        model.layers[i].name=model.layers[i].name + suffix
    return model

def KModel_to_sequential(model):    
    sqmodel = Sequential()
    for layer in model.layers[0:len(model.layers)-1]:
        sqmodel.add(layer)
        print(layer)
    return sqmodel

def KModel_resnet18_1x1combi3x3(INPUT_SHAPE=(224, 224, 3),OUTPUT_SHAPE=(10), pretrained='imagenet', top_layers=True):
    INPUT_SHAPE_RGB=(INPUT_SHAPE[0],INPUT_SHAPE[1],3)
    INPUT_SHAPE_BW=(INPUT_SHAPE[0],INPUT_SHAPE[1],3)
    rgbinput=Input(shape=INPUT_SHAPE_RGB)
    bwinput=Input(shape=INPUT_SHAPE_BW)
    
    #rgb 1x1 resnet18 model
    rgbmodel=KModel_resnet18(INPUT_SHAPE_RGB,OUTPUT_SHAPE,None,top_layers)
    rgbmodel=change_kernelsizes(rgbmodel,target_kernel_size=(1,1))
    #rgbmodel=change_strides(rgbmodel,target_strides=(1,1))
    rgbmodel=change_padding(rgbmodel,padding='same')
    rgbmodel.layers.pop(-1)
    #rgbmodel.layers.pop(-1)
    #rgbmodel_concatlayer=Flatten()(rgbmodel.layers[-1].output)
    resnet18_rgbmodel=Model(inputs=rgbmodel.input,outputs=rgbmodel.layers[-1].output)
    resnet18_rgbmodel=add_layers_name_suffix(resnet18_rgbmodel,"rgb")       
    
    
    #bw 3x3 resnet18 model
    bwmodel=KModel_resnet18(INPUT_SHAPE_BW,OUTPUT_SHAPE,pretrained,top_layers)
    #bwmodel=change_strides(bwmodel,target_strides=(1,1))
    bwmodel=change_padding(bwmodel,padding='same')
    bwmodel.layers.pop(-1)
    #bwmodel.layers.pop(-1)
    #bwmodel_concatlayer=Flatten()(bwmodel.layers[-1].output)
    resnet18_bwmodel = Model(inputs=bwmodel.input, outputs=bwmodel.layers[-1].output)
    resnet18_bwmodel=add_layers_name_suffix(resnet18_bwmodel,"bw")       

    #combined
    combined=Concatenate()([resnet18_rgbmodel.layers[-1].output, resnet18_bwmodel.layers[-1].output])
    combined=Dense(OUTPUT_SHAPE, activation="softmax")(combined)
    #combined=Activation('softmax')
    model=Model(inputs=[resnet18_rgbmodel.input,resnet18_bwmodel.input],outputs=combined)
    
    
    return model
