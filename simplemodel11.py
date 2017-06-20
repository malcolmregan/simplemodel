from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import os
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt
import numpy as np
import os

def initialize_model():
    model=Sequential()

    layers = [Conv2D(filters=8, kernel_size=5, strides=1, padding='same', input_shape=(28,28,1), activation='relu'),
              #MaxPooling2D(pool_size=2, strides=2, padding='same'),
              Conv2D(filters=8, kernel_size=5, strides=1, padding='same', activation='relu'),
              #MaxPooling2D(pool_size=2, strides=2, padding='same'),
              Flatten(),
              Dense(11)]

    for layer in layers:
        model.add(layer)

    model.add(Activation('softmax'))
    
    #dont know if compile is right - just copied and pasted from keras docs - MAKE SURE ITS RIGHT
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])   
 
    return model

def save_CPPN_model(model, iteration):
    save_path = './ckpt/CPPNx{}'.format(iteration)
    # If target directory does not exist, create
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Construct full path
    filepath = os.path.join(save_path, 'checkpoint.h5')
    
    model.save(filepath)
    print("Model was saved to: " + filepath)


def load_CPPN_model(iteration):
    load_path = './ckpt/CPPNx{}'.format(iteration)
    
    # Construct full path to dumped model
    filepath = os.path.join(load_path, 'checkpoint.h5')

    # Check if file exists
    assert os.path.exists(filepath)

    return keras.models.load_model(filepath)

def augmentdataset(iteration):
    X_train, Y_trai, X_test, Y_tes = data_mnist()
    Y_test = np.zeros((np.shape(Y_tes)[0], 11))
    for s in range(np.shape(Y_tes)[0]):
        temp=np.zeros((1,11))
        temp[0,0:10]=Y_tes[s]
        Y_test[s]=temp
    Y_train = np.zeros((np.shape(Y_trai)[0], 11))
    for s in range(np.shape(Y_trai)[0]):
        temp=np.zeros((1,11))
        temp[0,0:10]=Y_trai[s]
        Y_train[s]=temp

    numberoffiles=0
    for itera in range(1, iteration+1):
        for CLASS in range(10):
            numberoffiles = numberoffiles + len(os.walk('./CPPNGenerated/CPPNx{}/class{}'.format(itera, CLASS)).next()[2])

    examples = np.zeros([np.shape(X_train)[0]+numberoffiles, 28, 28, 1, 11])

    k=0
    for i in range(np.shape(X_train)[0]):
        examples[i,:,:,:,0] = X_train[i]
        examples[i,0,0,0,:][np.where(Y_train[i]==1)[0][0]] = 1
        k=k+1

    for itera in range(1,iteration+1):
        for CLASS in range(10):
            files_in_dir = os.listdir('./CPPNGenerated/CPPNx{}/class{}'.format(itera, CLASS))
            for files in files_in_dir:
                data = np.load(os.path.join('./CPPNGenerated/CPPNx{}/class{}'.format(itera, CLASS), files))
                array = data['features']
                array = array[0,0,:,:]
                examples[k,:,:,0,0] = array
                examples[k,0,0,0,:] = [0,0,0,0,0,0,0,0,0,0,1]
                k=k+1
   
    np.random.shuffle(examples)

    X_train = examples[:,:,:,:,0]
    Y_train = examples[:,0,0,0,:]
    return X_train, Y_train, X_test, Y_test
