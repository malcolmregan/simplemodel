from neat import nn, population, statistics
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from simplemodel11 import updatemodel

clazz = 0
k = 0
iteration = 1
global done
done = np.zeros((10),dtype=np.int8)
global gen
gen=0
global lastsave
lastsave=0
global highfit
highfit = 0
global lasthigh
lasthigh = 0
q=' '

def loadcheckpoint(iteration):
    print "Loading Checkpoint..."
    ckpt_path='./ckpt/CPPNx{}/checkpoint.ckpt'.format(iteration)
    saver = tf.train.Saver()
    saver.restore(sess,ckpt_path)
    print('CPPNx{} model restored...'.format(iteration)) 

def makenewdirectories(iteration):
    if not os.path.exists('./CPPNGenerated/CPPNx{}'.format(iteration)):
        print "Making New Directories..."
        os.makedirs('./CPPNGenerated/CPPNx{}'.format(iteration))
    for i in range(10):
        if not os.path.exists('./CPPNGenerated/CPPNx{}/class{}'.format(iteration, i)):
            os.makedirs('./CPPNGenerated/CPPNx{}/class{}'.format(iteration, i))

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

ckpt_path='./ckpt/CPPNx0/checkpoint.ckpt'
saver = tf.train.Saver()
saver.restore(sess,ckpt_path)

def softmax(g):
    e_x = np.exp(g - np.max(g))
    return e_x / e_x.sum()

#CPPN input
inp=[0]*(28*28)
n=0
for i in range(0, 28):
    for j in range(0, 28):
            inp[n]=(i,j)
            n=n+1

def get_fitness(g, inp, CLASS, k, iteration):
    global gen
    global lastsave
    global highfit
    global lasthigh
    global elitismcounter
    global done
    save_path='./CPPNGenerated/CPPNx{}/class{}'.format(iteration, CLASS)
    if pop.generation==0:
        gen=pop.generation
        lastsave = 0
        highfit = 0
        elitismcounter = 0
    if gen!=pop.generation:
        print "Class {}, CPPNx{}, Cycle {}\nSince last improvement: {}\nSince last save: {}\nElitism = {}\nDone: {}".format(CLASS,iteration,k,gen-lasthigh,gen-lastsave,pop.reproduction.elitism,done)
        gen=pop.generation
        if pop.reproduction.elitism == 0:
            elitismcounter=elitismcounter+1
            print elitismcounter
            if elitismcounter >= 10:
                pop.reproduction.elitism=1
                pop.config.compatibility_threshold=3.0
                elitismcounter=0
                lasthigh=gen
                num=0
                for key in pop.statistics.generation_statistics[gen-1]:
                    if np.max(pop.statistics.generation_statistics[gen-1][key])>num:
                        num=np.max(pop.statistics.generation_statistics[gen-1][key])
                highfit=num

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
    if fitness>highfit:
        highfit = fitness
        lasthigh = gen
        if pop.reproduction.elitism==0:
            pop.reproduction.elitism=1
            pop.config.compatibility_threshold=3.0
            elitismcounter=0
    if fitness>.9: 
        filename = "{}_{}_".format(k, g.ID)
        if os.path.isfile(os.path.join(save_path,filename+'.npz'))==False:
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
            #Save data
            if duplicate_flag==0:
                Z=np.zeros((1,1,28,28),dtype=np.float32)
                Z[0,0,:,:] = outputarray
                np.savez(os.path.join(save_path,filename+'.npz'),**{'features': Z, 'targets': [CLASS]})
                save_flag = 1
                lastsave = gen
            if fitness > 0.99 and save_flag == 1:
                fitness = 100
    if highfit < 0.15:
        eliteswitch = 50
    elif highfit >=0.15 and highfit<0.5:
        eliteswitch = 100
    elif highfit >=0.5 and highfit<0.8:
        eliteswitch = 200
    else:
        eliteswitch = 300
    if gen - lasthigh >= eliteswitch and pop.reproduction.elitism==1:
        pop.reproduction.elitism=0
        pop.config.compatibility_threshold=100.0
    if gen - lastsave > 3000:
        fitness = 100
    return fitness

gen_iter=0
for dirs in os.listdir('./CPPNGenerated/'):
    num=int(str(dirs).split('x')[1])
    if num>gen_iter:
        gen_iter=num
train_iter=0
for dirs in os.listdir('./ckpt/'):
    num=int(str(dirs).split('x')[1])
    if num>train_iter:
        train_iter=num
if gen_iter >= 1:
    if gen_iter > train_iter:
        q = raw_input('Continue [G]enerating for iteration {} or [T]rain next model? '.format(gen_iter))
    if train_iter == gen_iter:
        iteration = end_iter(gen_iter)


def end_iter(iteration):
    updatemodel(iteration)
    loadcheckpoint(iteration)
    iteration = iteration + 1
    makenewdirectories(iteration)
    return iteration

if q == 'G':
    iteration = gen_iter
    loadcheckpoint(iteration-1)
    k = 0
    for files in os.listdir('./CPPNGenerated/CPPNx{}/class0'.format(iteration)):
        high = int(str(files).split('_')[0])
        if high > k:
            k = high
    hask=np.zeros([10])
    for CLASS in range(10):
        for files in os.listdir('./CPPNGenerated/CPPNx{}/class{}'.format(iteration,CLASS)):
            q = int(str(files).split('_')[0])
            if k==q:
                hask[CLASS]=1
    clazz=(np.max(np.where(hask==1)[0])+1)%10
    if clazz==0:
        k=k+1
    
elif q == 'T':
    iteration = gen_iter
    iteration = end_iter(iteration)

def eval_fitness(genomes):
    for g in genomes:
        i = 0
        g.fitness = get_fitness(g, inp, clazz, k, iteration)

while 1:
    if len(os.listdir('./CPPNGenerated/CPPNx{}/class{}'.format(iteration, clazz)))<800:
        if done[clazz]<3:
            local_dir = os.path.dirname(__file__)
            config_path = os.path.join(local_dir, 'CPPN_config')
            pop = population.Population(config_path)
            pop.run(eval_fitness, 1000001)
    else:
        done[clazz]=3
    if gen - lastsave >= 3000:
        done[clazz] = done[clazz]+1
    if np.count_nonzero(done>=3) >= 8:
        iteration = end_iter(iteration)
        clazz = 9
        k = -1
        done = np.zeros((10),dtype=np.int8)
    if done[clazz]<3 and done[clazz]>0 and gen-lastsave<3000:
        done[clazz]=done[clazz]-1
    gen = 0
    lastsave = 0
    clazz = (clazz+1)%10
    if clazz == 0:
        k = k + 1
