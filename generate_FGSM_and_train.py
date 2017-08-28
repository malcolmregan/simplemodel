import tensorflow as tf
import numpy as np
import os
from fgsm_stuff import DO_FGSM, get_fgsm_data

from tensorflow.examples.tutorials.mnist import input_data
from simplemodel11 import initialize_model, save_CPPN_model, load_CPPN_model, augmentdataset

import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils import cnn_model
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval

# add extra class to MNIST example targets
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

sess = tf.Session()
keras.backend.set_session(sess)

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 11])

model=initialize_model() # intialize model located in simplemodel11.py 
predictions = model(x)

# train network on mnist
model.fit(X_train,Y_train,epochs=10,batch_size=128,verbose=1) # fit and evaluate are Keras functions
score = model.evaluate(X_test,Y_test,verbose=0)
print(score[1])

# perform FGSM and get FGSM examples (both of these functions located in fgsm_stuff.py)
DO_FGSM(model,x,y,predictions,sess,0)
X_train, Y_train = get_fgsm_data(0)
    
del model
del predictions

model = intialize_model()
predictions=model(x)

# train model on FGSM examples
model.fit(X_train,Y_train,epochs=10,batch_size=128,verbose=1)
score = model.evaluate(X_test,Y_test,verbose=0)
print(score[1])

os.remove('./FGSMGenerated/') # remove directory created by DO_FGSM()
