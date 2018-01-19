# Vanilla Feedforward Neural Network

Code Assignment for Ascent Robotics

## Synopsis

### Requirements
- The neural network implementation is developed under python 3.6.0 with numpy 1.14.0.
- The Iris example requires scikit-learn to load.
- Exact environment during development can be reproduced with `pip install -r requirements.txt`.

### Running the Iris example
Run `python3 iris_learn.py` with the working directory set as the parent directory of **nnet/**

## Files

### Main Directory
**iris_learn.py**
- Example usage of the neural network model on the Iris dataset.
- See previous section for instructions on how to run this file.

### nnet/
**activations.py**
- Contains activation functions.
- Currently available activation functions: sigmoid, softmax.

**optimizers.py**
- Contains optimization methods for training.
- Currently available optimizers: sgd, Adam.

**layers.py**
- Contains layers that construct the network.
- Currently available layers: Dense, Dropout.

**models.py**
- Contains the main neural network model.

## Report

### Design Considerations
The interface of the neural network was inspired by previous experiences with Keras. 

In order to maintain simplicity on the outermost level while retaining the flexibility to configure different aspects of the neural network, different activation functions, optimizers, and layers are created as modules that can then be imported to define the architecture of the neural network.

### Iris Dataset

### Performance Analysis


