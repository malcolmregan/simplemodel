from blocks.model import Model
from fuel.datasets.hdf5 import H5PYDataset
import matplotlib.pyplot as plt
import os
import numpy as np
import random

ITER_NUM=0
datadir='./FGSMGenerated/FGSMx{}/'.format(ITER_NUM)

for CLASS in range(0,10):
    fig = plt.figure()
    print 'Class {0}'.format(CLASS)

    imgdir = os.path.join(datadir, 'class{}'.format(CLASS))
    files_in_dir = os.listdir(imgdir)

    num_of_files=len(files_in_dir)
    print num_of_files

    examples=['']*num_of_files
    for i, files in enumerate(files_in_dir):
        examples[i] = os.path.join(imgdir,files)
    random.shuffle(examples)    

    if num_of_files>=25:
        ax = [0]*25
        for i in range(25):
            print i, examples[i]
        
            data=np.load(examples[i])
            array=data['features']
            array=array[0,0,:,:]
       
            ax[i] = fig.add_subplot(5 ,5, i+1, aspect='equal')
            ax[i].imshow(array, cmap='Greys', interpolation='nearest')

    else:
        ax = [0]*num_of_files
        for i in range(num_of_files):
            print i, examples[i]

            data=np.load(examples[i])
            array=data['features']
            array=array[0,0,:,:]

            ax[i] = fig.add_subplot(5 ,5, i+1, aspect='equal')
            ax[i].imshow(array, cmap='Greys', interpolation='nearest')

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    plt.show()
    #plt.suptitle('CPPNx{}, Class {}'.format(ITER_NUM, CLASS), size=25)
    #plt.savefig('CPPNGenerated/images/CPPNx{}/Class{}.png'.format(ITER_NUM, CLASS))
    #plt.close()
