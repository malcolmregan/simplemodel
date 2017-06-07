### 2 convolutional layers, 1 fully-connected layer for MNIST

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

sess=tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 11])

examples = np.zeros([np.shape(mnist.train.images)[0],np.shape(mnist.train.images)[1], 11])
k=0
for i in range(np.shape(mnist.train.images)[0]):
    examples[i,:,0] = mnist.train.images[i]
    examples[i,0,:][np.where(mnist.train.labels[i]==1)[0][0]] = 1
    k=k+1

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

W_fc1 = weight_variable([392,11]) ###392 by big number and then another fully connected layer??
b_fc1 = bias_variable([11])
h_pool2_flat = tf.reshape(h_pool2, [-1, 392])

y_conv = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

batchcount=0
for i in range(20000):
    if (batchcount+50)%np.shape(examples)[0]<batchcount%np.shape(examples)[0]:
        batchcount=0
    xbatch = examples[(batchcount%np.shape(examples)[0]):((batchcount+50)%np.shape(examples)[0]),:,0]
    ybatch = examples[(batchcount%np.shape(examples)[0]):((batchcount+50)%np.shape(examples)[0]),0,:]
    if i%100==0:
        train_accuracy = accuracy.eval(feed_dict={x: xbatch, y_: ybatch})
        print("Step: %d, Training Accuracy: %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: xbatch, y_: ybatch})
    batchcount=batchcount+50

testlabels = np.zeros([np.shape(mnist.test.labels)[0], 11])
for k in range(np.shape(mnist.test.labels)[0]):
    testlabels[k] = np.append(mnist.test.labels[k],0)

print("Test Accuracy: %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: testlabels}))

saver = tf.train.Saver()
snapshot_name = "checkpoint"
fn = saver.save(sess, "%s/%s.ckpt" % ('./ckpt/CPPNx0', snapshot_name))
print("Model saved in file: %s" % fn)
