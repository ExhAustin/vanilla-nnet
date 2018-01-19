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
#       sigmoid, softmax
#   - Optimizers available: sgd, Adam
#   - Loss functions available: quadratic 
#       (coded inherently, not adjustable in this primitive version)
#
#####################################################################

import numpy as np

class NeuralNet:
    def __init__(self, optimizer, lr):
        self.n_layers = 0
        self.layers = []
        self.optimizer_str = optimizer
        self.lr = lr

    # Add layer to network
    def add_layer(self, layer):
        self.layers.append(layer)
        layer.init_optimizer(self.optimizer_str, self.lr)
        self.n_layers += 1

    # Predict labels
    def predict(self, X, istrain=False):
        output = X
        for layer in self.layers:
            assert (output.shape[1] == layer.n_in), 'Layer dimensions does not match'
            output = layer.forward(output, istrain)

        return output
    
    # Train on single batch
    def train_on_batch(self, X, Y):
        # Forward propagation
        Y_pred = self.predict(X, istrain=True)

        # Loss function (Quadratic)
        err = Y_pred - Y
        mse = np.mean(err**2)

        # Backpropagation
        err_signal = 2*err
        for layer in self.layers[::-1]:
            # Propagate errors
            err_signal, grads = layer.backprop(err_signal)

            # Update layer weights
            layer.update(grads)

        # Return mean squared error
        return mse

    # Train on entire dataset
    def train(self, X, Y, max_epochs, batchsize=1, validation_rate=0, 
            early_stop=False, verbose=True):
        n_data = Y.shape[0]
        n_train_data = int(np.floor(n_data * (1-validation_rate)))
        n_batches = int(np.ceil(n_train_data/batchsize))
        prev_err = None

        if verbose:
            print('Training network...')

        # Iteratively train network
        for e in range(max_epochs):
            # Train epoch
            total_mse = 0
            for i in range(n_batches):
                # Sample batch
                s = np.random.randint(0, n_train_data, batchsize)
                X_batch = X[s,:]
                Y_batch = Y[s,:]

                # Train batch
                mse = self.train_on_batch(X_batch, Y_batch)
                total_mse += mse/n_batches

            # Validate
            if validation_rate:
                Y_pred = self.predict(X[n_train_data:,:])
                err = Y[n_train_data:,:] - Y_pred
                validate_mse = np.mean(err**2)

            # Output log
            if verbose:
                print('Epoch ', e+1, ':', end=' \t')
                print('mse=', total_mse, end=', \t')
                if validation_rate:
                    print('validate_mse=', validate_mse, end=', \t')
                print()

            # Early stopping
            if validation_rate and early_stop:
                if prev_err is None or validate_mse < prev_err:
                    prev_err = validate_err
                else:
                    break
                    if verbose:
                        print('Training terminated due to increasing validation errors.')

        # Training completed
        if verbose:
            print('Training completed.')
            if validation_rate:
                print('Final validation error: ', validate_mse)

        # Return final error
        return total_mse
