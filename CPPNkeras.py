from neat import nn, population, statistics
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

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval

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

if not os.path.exists('./ckpt'):
    os.makedirs('./ckpt')

if not os.path.exists('./CPPNGenerated'):
    os.makedirs('./CPPNGenerated')

if not os.path.exists('./ckpt/CPPNx0/'):
    model=initialize_model()
    predictions = model(x)

    # train network on mnist
    model.fit(X_train,Y_train,epochs=10,batch_size=128,verbose=1)
    score = model.evaluate(X_test,Y_test,verbose=0)
    print(score[1])

    # perform FGSM and get FGSM examples
    DO_FGSM(model,x,y,predictions,sess,0)
    X_train, Y_train, X_test, Y_test =  augmentdataset(0) # FYI: augmentdataset() just returns MNIST testset as it was before, only training set is augmented
   
    del model
    del predictions
    model = initialize_model()
    predictions=model(x)
    score = model.evaluate(X_test,Y_test,verbose=0)
    print(score[1])    

    # train model on FGSM examples
    model.fit(X_train,Y_train,epochs=10,batch_size=128,verbose=1)
    score = model.evaluate(X_test,Y_test,verbose=0)
    print(score[1])
   
    save_CPPN_model(model,0)

def loadcheckpoint(iteration):
    print "Loading Checkpoint..."
    model = load_CPPN_model(iteration)
    return model

def makenewdirectories(iteration):
    if not os.path.exists('./CPPNGenerated'):
        os.makedirs('./CPPNGenerated')
    if not os.path.exists('./CPPNGenerated/CPPNx{}'.format(iteration)):
        print "Making New Directories..."
        os.makedirs('./CPPNGenerated/CPPNx{}'.format(iteration))
    for i in range(10):
        if not os.path.exists('./CPPNGenerated/CPPNx{}/class{}'.format(iteration, i)):
            os.makedirs('./CPPNGenerated/CPPNx{}/class{}'.format(iteration, i))

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
    
    feat = np.zeros((1,28,28,1))
    feat[0,:,:,0]=outputarray

    pred = predictions.eval(session=sess, feed_dict={x: feat})[0]
 
    pred[:CLASS]=np.absolute(pred[:CLASS])     
    pred[(CLASS+1):]=np.absolute(pred[(CLASS+1):])
    if np.sum(pred-min(pred))==0:
        fitness=0
    if np.sum(pred-min(pred))!=0:
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
    if gen > 5000:
        fitness = 100
        lastsave = gen+3000
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
        iteration = gen_iter+1
        makenewdirectories(iteration)    
   
if q == 'G':
    iteration = gen_iter
    model = loadcheckpoint(iteration-1)
    predictions = model(x)
    k = 0
    for CLASS in range(10):
        for files in os.listdir('./CPPNGenerated/CPPNx{}/class{}'.format(iteration,CLASS)):
            high = int(str(files).split('_')[0])
            if high > k:
                k = high
    hask=np.zeros([10])
    for CLASS in range(10):
        for files in os.listdir('./CPPNGenerated/CPPNx{}/class{}'.format(iteration,CLASS)):
            q = int(str(files).split('_')[0])
            if k==q:
                hask[CLASS]=1
    if len(np.where(hask==1)[0])>0:
        clazz=(np.max(np.where(hask==1)[0])+1)%10
        if clazz==0:
            k=k+1
    if len(np.where(hask==1)[0])==0:
        clazz=0
    
elif q == 'T':
    iteration = gen_iter
    X_train, Y_train, X_test, Y_test = augmentdataset(iteration)
    model = initialize_model()
    predictions = model(x)

    model.fit(X_train,Y_train,epochs=10,batch_size=128,verbose=1)
    score = model.evaluate(X_test,Y_test,verbose=0)
    print(score[1])
    
    DO_FGSM(model,x,y,predictions,sess,iteration)

    X_train, Y_train, X_test, Y_test = augmentdataset(iteration)
    del model
    del predictions
    model = initialize_model()
    predictions = model(x)
    model.fit(X_train,Y_train,epochs=10,batch_size=128,verbose=1)
    score = model.evaluate(X_test,Y_test,verbose=0)
    print(score[1])

    save_CPPN_model(model,iteration)
    iteration=iteration+1
    makenewdirectories(iteration)


def eval_fitness(genomes):
    for g in genomes:
        i = 0
        g.fitness = get_fitness(g, inp, clazz, k, iteration)

while 1:
    if not os.path.exists('./CPPNGenerated/CPPNx1'):
        makenewdirectories(1)
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
    if np.count_nonzero(done>=3) == 10: # for later iterations mayeb change to count_nonxero>=8

        X_train, Y_train, X_test, Y_test = augmentdataset(iteration)
        del model
        del predictions
        model = initialize_model()
        predictions = model(x)
        model.fit(X_train,Y_train,epochs=10,batch_size=128,verbose=1)
        score = model.evaluate(X_test,Y_test,verbose=0)
        print(score[1])

        DO_FGSM(model,x,y,predictions,sess,iteration)

        X_train, Y_train, X_test, Y_test = augmentdataset(iteration)
        del model
        del predictions
        model = initialize_model()
        predictions = model(x)
        model.fit(X_train,Y_train,epochs=10,batch_size=128,verbose=1)
        score = model.evaluate(X_test,Y_test,verbose=0)
        print(score[1])

        save_CPPN_model(model,iteration)
        iteration=iteration+1
        makenewdirectories(iteration)

        raw_input()
        clazz = 9
        k = -1
        done = np.zeros((10),dtype=np.int8)
    if done[clazz]<3 and done[clazz]>0 and gen-lastsave<3000:
        done[clazz]=done[clazz]-1
    gen = 0
    lastsave = 0
    lasthigh = 0
    clazz = (clazz+1)%10
    if clazz == 0:
        k = k + 1
