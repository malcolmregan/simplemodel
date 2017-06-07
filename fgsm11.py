import tensorflow as tf
import numpy as np
from cleverhans.attacks import FastGradientMethod
from tensorflow.examples.tutorials.mnist import input_data

ITER_NUM = 2

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 11])

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
W_conv1 = weight_variable([5,5,1,8])
b_conv1 = bias_variable([8])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = maxpool(h_conv1)
W_conv2 = weight_variable([5,5,8,8])
b_conv2 = bias_variable([8])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = maxpool(h_conv2)
W_fc1 = weight_variable([392,11])
b_fc1 = bias_variable([11])
h_pool2_flat = tf.reshape(h_pool2, [-1, 392])
y_conv = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    
#Restore checkpoint, start session
ckpt_path='./ckpt/CPPNx{0}/checkpoint.ckpt'.format(ITER_NUM)
saver = tf.train.Saver()
saver.restore(sess,ckpt_path)

val = mnist.train.next_batch(1000)
x_val = val[0]
y_val = val[1]

def model(inp):
    return tf.nn.softmax(y_conv)

fgsm = FastGradientMethod(model, 'tf', sess=sess)
fgsm_params = {'eps': 0.3} #see what putting in labels does
adv_x = fgsm.generate(x, **fgsm_params)
example = sess.run(adv_x, feed_dict={x: x_val})
preds_adv = model(adv_x)
prediction = sess.run(preds_adv, feed_dict={x: example})

import matplotlib.pyplot as plt

for CLASS in range(10):
    fig = plt.figure()
    ax=[0]*25
    count = 0
    pred=0
    for i in range(1000):
        if prediction[i][CLASS]>pred:
            pred=prediction[i][CLASS]
        if count<25:
            if prediction[i][CLASS]>.9:
                temp = np.reshape(example[i], [28,28])
                ax[count] = fig.add_subplot(5 ,5, count+1, aspect='equal')
                ax[count].imshow(temp, cmap='Greys', interpolation='nearest')
                count=count+1
    print CLASS, pred

    #plt.show()
    #plt.suptitle('FGSM for CPPNx{0}, Class {1}'.format(ITER_NUM, CLASS), size=25)
    #plt.savefig('FGSMGenerated/images/CPPNx{0}/Class{1}.png'.format(ITER_NUM, CLASS))
    #plt.close()

