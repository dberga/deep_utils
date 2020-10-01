
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display #display plots in line
import pickle as pkl #to save/load files


        

def show_batch(image_batch, label_batch, CLASS_NAMES):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      if np.shape(image_batch[n])[2]==1: #channels are b/w    
          plt.imshow(np.reshape(image_batch[n],[np.shape(image_batch[n])[0],np.shape(image_batch[n])[1]]))
      else: #channels are rgb    
          plt.imshow(image_batch[n])
      #plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.title(str(CLASS_NAMES[label_batch[n]]))
      plt.axis('off')
  #plt.show()
  plt.pause(1)
  plt.close()



import tensorflow as tf
class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
        
        
def plot_all_history(MODELS_LIST,LEGEND=None):
    if LEGEND is None:
        LEGEND=MODELS_LIST
        
    for m in range(len(MODELS_LIST)):
        try:
            with open("history_%s.pkl"%MODELS_LIST[m],'rb') as f:
                history=pkl.load(f)
                plt.plot(history['acc'],label=LEGEND[m])
        except:
            try:
                with open("history_%s.pkl"%MODELS_LIST[m],'rb') as f:
                    history=pkl.load(f)
                    plt.plot(history['accuracy'],label=LEGEND[m])
            except:
                print("Please run again %s"%MODELS_LIST[m])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('model accuracy (train)')
    plt.legend(loc="upper left")
    plt.savefig('accuracy_train.png')
    plt.close()
    
    for m in range(len(MODELS_LIST)):
        try:
            with open("history_%s.pkl"%MODELS_LIST[m],'rb') as f:
                history=pkl.load(f)
                plt.plot(history['val_acc'],label=LEGEND[m])
        except:
            try:
                with open("history_%s.pkl"%MODELS_LIST[m],'rb') as f:
                    history=pkl.load(f)
                    plt.plot(history['val_accuracy'],label=LEGEND[m])
            except:
                print("Please run again %s"%MODELS_LIST[m])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('model accuracy (val)')
    plt.legend(loc="upper left")
    plt.savefig('accuracy_val.png')
    plt.close()
    
    for m in range(len(MODELS_LIST)):
        try:
            with open("history_%s.pkl"%MODELS_LIST[m],'rb') as f:
                history=pkl.load(f)
                plt.plot(history['loss'],label=LEGEND[m])
        except:
            print("Please run again %s"%MODELS_LIST[m])
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('model loss (train)')
    plt.legend(loc="upper left")
    plt.savefig('loss_train.png')
    plt.close()
    
    for m in range(len(MODELS_LIST)):
        try:
            with open("history_%s.pkl"%MODELS_LIST[m],'rb') as f:
                history=pkl.load(f)
                plt.plot(history['val_loss'],label=LEGEND[m])
        except:
            print("Please run again %s"%MODELS_LIST[m])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('model loss (val)')
    plt.legend(loc="upper left")
    plt.savefig('loss_val.png')
    plt.close()





