#####################################################################
# A Simple Neural Network Model
#####################################################################
#
# Author:   Austin Shih-Ping Wang
# Year:     2018
#
# Notes:
#   - Input and output arrays are 2d numpy arrays
#   - Columns: Features
#   - Rows: Data entries
#   - Layers available: Dense, Dropout
#   - Activation functions available: 
#       linear, sigmoid, tanh, relu, softmax
#   - Optimizers available: sgd, Adam
#   - Loss functions available: quadratic 
#       (coded inherently, not adjustable in this primitive version)
#
#####################################################################

import numpy as np
import .optimizers

def NeuralNet:
    def __init__(self):
        self.n_layers = 0
        self.layers = []

    # Add layer to network
    def add_layer(self, layer):
        self.layers.append(layer)
        self.n_layers += 1

    # Predict labels
    def predict(self, X, istrain=False):
        output = X
        for layer in self.layers:
            assert (output.shape[1] == layer.n_in), 'Layer dimensions does not match'
            output = layer.forward(output, istrain)

        return output
    
    # Train network
    def train(self, X, Y):
        # Forward propagation
        Y_pred = self.predict(X, istrain=True)

        # Loss function (Quadratic)
        err = (Y_pred - Y)
        mse = np.mean(np.dot(err, err.T))

        # Backpropagation
        err_prop = 2*err
        for layer in self.layers[::-1]:
            # Propagate errors
            err_prop, w_grad, b_grad = layer.backprop(err_prop)

            # Update layer weights

        # Return mean squared error
        return mse

