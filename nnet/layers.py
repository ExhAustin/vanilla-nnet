# Layer types for Neural Networks

import numpy as np
import .activations
import .optimizers

def Dense:
    def __init__(self, n_in, n_out, activation):
        # Initialize parameters
        episilon = 1e-7
        self.w = epsilon*np.random.randn(n_in, n_out) 
        self.b = epsilon*np.random.randn(1, n_out) 

        # Define activation function
        assert (activation in activations.function_list), 'Invalid activation function.'
        self.sigma = np.eval('activations.' + activation + '()')

    # Forward evaluation
    def forward(self, x, istrain=True):
        self.x = x
        self.z = np.dot(self.x, self.w) + self.b
        return self.sigma.forward(self.z)

    # Partial gradient backpropagation (da/dw | x)
    def backprop(self, err):
        dsigma = self.sigma.gradient(self.z)
        b_grad = err * dsigma
        w_grad = np.dot(x.T, b_grad)
        err_prop = np.dot(err * dsigma, self.w.T)

        return err_prop, w_grad, b_grad

    # Update weights
    def update(self, grads, optimizer):
        for self.w


def Dropout:
    def __init__(self, n_in, rate):
        self.n = n_in
        self.rate = rate    # dropout rate
        self.mask = np.ones(self.n)

    # Forward evaluation
    def forward(self, x, istrain=True):
        if istrain:
            self.resample()
            return x*self.mask
        else:
            return x*(1-self.rate)

    # Partial gradient backpropagation
    def backprop(self):
        return self.mask

    # Update weights
    def update(self, grads, optimizer):
        return  # No weights to update for dropout layers

    # Resample mask
    def resample(self):
        self.mask = (np.random.uniform(size=self.n) > self.rate).astype('float')
