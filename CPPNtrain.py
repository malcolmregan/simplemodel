from neat import nn, population, statistics
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

x = tf.placeholder(tf.float32, shape=[784])
y_ = tf.placeholder(tf.float32, shape=[11])

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
ckpt_path='./ckpt/checkpointtest.ckpt'
saver = tf.train.Saver()
saver.restore(sess,ckpt_path)

def softmax(g):
    e_x = np.exp(g - np.max(g))
    return e_x / e_x.sum()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

#CPPN input
inp=[0]*(28*28)
n=0
for i in range(0, 28):
    for j in range(0, 28):
            inp[n]=(i,j)
            n=n+1

def get_fitness(g, inp, CLASS, k):
    global gen
    save_path='./CPPNGenerated/CPPNtrain/class{}'.format(CLASS)
    if pop.generation==0:
        gen=pop.generation
    if gen!=pop.generation:
        print "Class: {}".format(CLASS)
        gen=pop.generation

    net = nn.create_feed_forward_phenotype(g)
    outputarray = [0]*28*28
    for inputs in inp:
        output = net.serial_activate(inputs)
        outputarray[inputs[0]+inputs[1]*28] = output[0]

    outputarray = np.reshape(outputarray,(28,28))
    
    feat = np.ndarray.flatten(np.asarray(outputarray, dtype=np.float32))

    pred = y_conv.eval(session=sess, feed_dict={x: feat})[0]    
    pred = softmax(pred)   

    pred[:CLASS]=np.absolute(pred[:CLASS])     
    pred[(CLASS+1):]=np.absolute(pred[(CLASS+1):])
    fitness = (pred[CLASS]-min(pred))/(np.sum(pred-min(pred)))
    if fitness>.98: 
        filename = "{}_{}_".format(k, g.ID)
        if os.path.isfile(os.path.join(save_path,filename+'.npz'))==False:
            # Check for duplicates - slow come up with a better way
            files_in_dir = os.listdir(save_path)
            duplicate_flag=0
            save_flag=0
            for files in files_in_dir:
                if duplicate_flag == 0:
                    data=np.load(os.path.join(save_path, files))
                    array=data['features']
                    array=array[0,0,:,:]
                    diff = np.sum(np.absolute(outputarray-array))
                    if diff < 100:
                        duplicate_flag=1
                        #fitness = 0 #?

            if duplicate_flag==0:
                #Train model
                batch = feat
                train_step.run(session=sess, feed_dict={x: batch, y_: np.asarray([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.])})

                #Save data
                Z=np.zeros((1,1,28,28),dtype=np.float32)
                Z[0,0,:,:] = outputarray
                np.savez(os.path.join(save_path,filename+'.npz'),**{'features': Z, 'targets': [CLASS]})
                save_flag=1
         
            if fitness>0.9999 and save_flag==1:
                fitness=100 
    return fitness

clazz=1
k=0

def eval_fitness(genomes):
    for g in genomes:
        i=0
        g.fitness = get_fitness(g, inp, clazz, k)
while 1:
    if len(os.walk('./CPPNGenerated/CPPNtrain/class{}'.format(clazz)).next()[2])<1000:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'CPPN_config')
        pop = population.Population(config_path)
        pop.run(eval_fitness, 1000001)
    clazz=(clazz+1)%10
    if clazz==0:
        k=k+1
