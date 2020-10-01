# -*- coding: utf-8 -*-

###################test each batch separately

if TEST==True:
    if BATCH_SIZE==-1:
        if mod=='bwmodel':
            test_loss, test_acc = bwmodel.evaluate(x=x_test2,  y=y_test, verbose=2)
            print('\nTest accuracy:', test_acc)
        if mod=='rgbmodel':
            test_loss, test_acc = rgbmodel.evaluate(x=x_test,  y=y_test, verbose=2)
            print('\nTest accuracy:', test_acc)
        if mod=='rgbmodel_3by3':
            test_loss, test_acc = rgbmodel_3by3.evaluate(x=x_test,  y=y_test, verbose=2)
            print('\nTest accuracy:', test_acc)
        if mod=='combi':
            test_loss, test_acc = combimodel.evaluate(x=[x_test,x_test2],  y=y_test, verbose=2)
            print('\nTest accuracy:', test_acc)
    else:
        for epoch in range(EPOCHS):
            for batch in np_test_dataset:
                images, labels = batch[0], batch[1]      
                test_loss, test_acc = rgbmodel.evaluate(images,  labels)
                print('\nTest accuracy:', test_acc)


###################train each batch separately

    history = combimodel.fit_generator(np_train_dataset,epochs=EPOCHS,steps_per_epoch=steps_per_epoch,verbose=2)
    
    #for epoch in range(EPOCHS):
        #for batch in np_train_dataset:
        #    images, labels = batch[0], batch[1]        
        #    rgbmodel.train_on_batch(images, labels)
    
    
##################clear session
        
from tensorflow.keras.backend import clear_session

def keras_clear_session():
    clear_session()
    
    
    ################### 
    
#tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir='log.log', profile_batch=0)]
        steps_per_epoch=np.ceil(num_training_samples/BATCH_SIZE)
    
    ################

if BATCH_SIZE==-1:
    x_train,y_train=np_train_dataset[0],np_train_dataset[1]
    x_test,y_test=np_test_dataset[0],np_test_dataset[1]
    x_val,y_val=np_val_dataset[0],np_val_dataset[1]
    
    x_train1=np_train_dataset[0]
    x_test1=np_test_dataset[0]
    x_val1=np_val_dataset[0]
    x_train2=np_train_dataset2[0]
    x_test2=np_test_dataset2[0]
    x_val2=np_val_dataset2[0]
else:
    steps_per_epoch=np.ceil(num_training_samples/BATCH_SIZE)
    
    
    