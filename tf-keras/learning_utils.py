# -*- coding: utf-8 -*-

from keras import optimizers
import pdb

def SelectOptimizer(opt='SGD', LR=0.01,multipliers=None):
    if opt=='SGD':
        return optimizers.SGD(learning_rate=LR,momentum=0.9,decay=0.01,nesterov=False)
    elif opt=='Adam':
        return optimizers.Adam(learning_rate=LR,beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.01,amsgrad=False)
    elif opt=='Adagrad':
        return optimizers.Adagrad(learning_rate=LR, epsilon=1e-08,decay=0.01)
    elif opt=='Adadelta':
        return optimizers.Adadelta(learning_rate=LR,rho=0.95, epsilon=1e-08,decay=0.01)
    elif opt=='RMSprop':
        return optimizers.RMSprop(learning_rate=LR,rho=0.9, epsilon=1e-08,decay=0.01)
    elif opt=='LR_SGD':
        #return optimizers.LearningRateMultiplier(optimizers.SGD,lr_multiplier=multipliers,momentum=0.9,nesterov=False)
        return LR_SGD(learning_rate=LR,momentum=0.9,decay=0.01,nesterov=False,multipliers=multipliers)
    else:
        return 'adam'

 

def getMultipliers(model,initLR=0.01,finLR=0.04): #minLR is initial LR
    trainable_layers=[]
    for i in range(len(model.layers)):
        '''
        try: #see if has 'trainable' key
            if model.layers[i].get_config()['trainable'] is True: #see if is trainable
                trainable_layers.append(i)
                print(model.layers[i].name)
        except:
            pass
        '''
        if "conv" in model.layers[i].name.lower() or "dense" in model.layers[i].name.lower():
            trainable_layers.append(i)
            
    multipliers={}
    if len(trainable_layers) is not 0:
        for j in range(len(trainable_layers)):
            factor=(((finLR/initLR)-1)/((len(trainable_layers)+1)/(j+1)))+1
            #print(factor)
            multipliers[model.layers[trainable_layers[j]].name]=factor
    print(multipliers)
    return multipliers
        
    

from keras.optimizers import Optimizer
from keras.legacy import interfaces

import keras.backend as K

class LR_SGD(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, learning_rate=0.01, momentum=0., decay=0.,
                 nesterov=False,multipliers=None,**kwargs):
        super(LR_SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.lr_multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate = learning_rate * (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            
            matched_layer = [x for x in self.lr_multipliers.keys() if x in p.name]
            if matched_layer:
                new_lr = learning_rate * self.lr_multipliers[matched_layer[0]]
            else:
                new_lr = learning_rate

            v = self.momentum * m - new_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - new_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(LR_SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 

    
   
