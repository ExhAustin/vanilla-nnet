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

## Assignment Report

### Design Considerations
The interface of the neural network was inspired by previous experiences with Keras. 

In order to maintain simplicity on the outermost level while retaining the flexibility to configure different aspects of the neural network, different activation functions, optimizers, and layers are created as modules that can then be imported to define the architecture of the neural network.

### Learning the Iris Dataset
Dataset information:
- Number of features: 4
- Classification categories: 3
- Data entries: 150

#### Data Preprocessing
The features are normalized to zero mean and unit variance so the features possess equal influence. Labels were originally given as category numbers, so we create an array of length 3 for each entry as a representation of the individual probabilities of each category, with a value of 1 as the element corresponding to the label and 0 for the others.



#### Neural Network Design
With only 4 features and 3 categories, layers do not need to be too wide, and since we are using logistic sigmoid activation functions, networks with too many layers will suffer from vanishing gradients severly. Thus, the neural network configuration chosen has two hidden layers with 64 and 16 neurons respectively. 

To evaluate the probabilities of each of the 3 categories, we use a softmax activation function for the output layer. The advantage of softmax over sigmoid is that it respects the fact that all probabilites add up to 1, so learning a decreased probability in one category translates to increased probabilities in other categories.

Dropout is added to the hidden layers, and the Adam optimizer is chosen for

### Performance Analysis


