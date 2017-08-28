from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import numpy as np
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import cnn_model
from simplemodel11 import load_CPPN_model

# change to create 800 examples that succesfully trick model for each class and save them to FGSMGenerated

def DO_FGSM(model, x, y, predictions, sess, ITER_NUM):
     
    if not os.path.exists('./FGSMGenerated'):
        os.makedirs('./FGSMGenerated/')
    if not os.path.exists('./FGSMGenerated/FGSMx{}'.format(ITER_NUM)):
        os.makedirs('./FGSMGenerated/FGSMx{}'.format(ITER_NUM))
    for i in range(10):
        if not os.path.exists('./FGSMGenerated/FGSMx{}/class{}'.format(ITER_NUM, i)):
            os.makedirs('./FGSMGenerated/FGSMx{}/class{}'.format(ITER_NUM, i))

    FLAGS = flags.FLAGS

    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
    np.set_printoptions(threshold=np.nan)

    """
    MNIST cleverhans tutorial
    :return:
    """
    keras.layers.core.K.set_learning_phase(0)

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend sessi
    #sess = tf.Session()
    #keras.backend.set_session(sess)

    # Get MNIST test data
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

    ## Find out what this does
    assert Y_train.shape[1] == 11.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    #x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    #y = tf.placeholder(tf.float32, shape=(None, 11))

    # Define TF model graph
    #model = load_CPPN_model(ITER_NUM)
    #predictions = model(x)
    #print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate test examples: %0.4f' % accuracy)

    # Train an MNIST model
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    #model_train(sess, x, y, predictions, X_train, Y_train,
    #            evaluate=evaluate, args=train_params)

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {'eps': 0.3}
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model(adv_x)

    X_train=X_test
    Y_train=Y_test
   
    count = 0
    print("Generating FGSM data")
    for idx in range(10000):
        examp = adv_x.eval(session=sess, feed_dict={x: X_train[idx:idx+1]})
        pred = predictions.eval(session=sess, feed_dict={x: examp})
        if np.where(pred[0]==max(pred[0]))[0] > .8:
            if np.where(pred[0]==max(pred[0]))[0][0] != np.where(Y_train[idx]==max(Y_train[idx]))[0][0]:
                save_path = './FGSMGenerated/FGSMx{}/class{}'.format(ITER_NUM, np.where(Y_train[idx]==max(Y_train[idx]))[0][0])
                filename = '{}'.format(count)
                Z=np.zeros((1,1,28,28),dtype=np.float32)
                Z[0,0,:,:] = examp[0,:,:,0]
                np.savez(os.path.join(save_path,filename+'.npz'),**{'features': Z, 'targets':  np.where(Y_train[idx]==max(Y_train[idx]))[0][0]})
                count = count + 1
    print(count)

def get_fgsm_data(ITER_NUM):
    numberoffiles = 0
    for CLASS in range(10):
        numberoffiles = numberoffiles + len(os.walk('./FGSMGenerated/FGSMx{}/class{}'.format(ITER_NUM, CLASS)).next()[2])

    examples = np.zeros([numberoffiles, 28, 28, 1, 11])

    k=0
    for CLASS in range(10):
        files_in_dir = os.listdir('./FGSMGenerated/FGSMx{}/class{}'.format(ITER_NUM, CLASS))
        for files in files_in_dir:
            data = np.load(os.path.join('./FGSMGenerated/FGSMx{}/class{}'.format(ITER_NUM, CLASS), files))
            array = data['features']
            targ = data['targets']
            array = array[0,0,:,:]
            examples[k,:,:,0,0] = array
            examples[k,0,0,0,:] = targ
            k=k+1

    np.random.shuffle(examples)

    X = examples[:,:,:,:,0]
    Y = examples[:,0,0,0,:]
    return X, Y

