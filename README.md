# simplemodel

The file generate_FGSM_and_train.py was created for the purposes of troubleshooting the problem at hand.
It trains a model on MNIST. generates FGSM examples for this model, and trains a new model on the FGSM examples.

TO RUN FGSM+CPPN TRAINING SCRIPT:

    add cleverhans path to PYTHONPATH before running:
    >>export PYTHONPATH=./cleverhans:$PYTHONPATH
    to run:    
    >>python CPPNkeras.py
    

File descriptions:

    CPPNkeras.py - main file. Runs iterative FGSM+CPPN generation and training for a simple model.
    CPPN_config - contains parameters and parameter values used by NEAT.
    fgsm_stuff.py - contains functions for generation and loading of FGSM examples.
    jsma.py - (another cleverhans method) not currently in use.
    plot25.py - for veiwing FGSM or CPPN examples. generates figures showing 25 CPPN or FGSM examples. 1 figure per class.
                must go into code and manually change example path if you want to plot CPPN examples. Must also manually 
                specify the iteration num.
    simplemodel11.py - contains functions to initialize, load, and save models. also contains augment_dataset function.
