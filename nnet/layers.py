# Layer types for Neural Networks

import numpy as np
import nnet.activations as activations
import nnet.optimizers as optimizers

class Dense:
    def __init__(self, n_in, n_out, activation):
        # Initialize parameters
        epsilon = 1e-2
        self.n_in = n_in
        self.n_out = n_out
        self.w = epsilon*np.random.randn(n_in, n_out) 
        self.b = epsilon*np.random.randn(1, n_out) 

        # Define activation function
        assert (activation in activations.function_list), 'Invalid activation function.'
        self.sigma = eval('activations.' + activation + '()')

    # Get output using forward evaluation
    def forward(self, x, istrain=False):
        self.x = x
        self.z = np.dot(self.x, self.w) + self.b
        return self.sigma.forward(self.z)

    # Get gradients using backpropagation
    def backprop(self, err):
        err_z = self.sigma.gradient(err, self.z, repeat=True)
        b_grad = np.sum(err_z, axis=0).reshape(1,-1)
        w_grad = np.dot(self.x.T, err_z)
        err_next = np.dot(err_z, self.w.T)
        #print('err_z=', err_z)
        #print('self.x', self.x)
        #print('self.w=', self.w)
        #print('err_next', err_next)
        #print('w_grad', w_grad)

        return err_next, [w_grad, b_grad]

    # Initialize optimizer
    def init_optimizer(self, optimizer, lr):
        assert (optimizer in optimizers.method_list), 'Invalid optimizer argument.'
        self.w_optimizer = eval('optimizers.' + optimizer + '(' + str(lr) + ')')
        self.b_optimizer = eval('optimizers.' + optimizer + '(' + str(lr) + ')')

    # Update weights using optimizer
    def update(self, grads):
        self.w = self.w_optimizer.update_weights(self.w, grads[0])
        self.b = self.b_optimizer.update_weights(self.b, grads[1])


class Dropout:
    def __init__(self, n_in, rate):
        self.n_in = n_in
        self.rate = rate    # dropout rate
        self.mask = np.ones(n_in)

    # Forward evaluation
    def forward(self, x, istrain=False):
        if istrain:
            self.resample()
            return x*self.mask
        else:
            return x*(1-self.rate)

    # Partial gradient backpropagation
    def backprop(self,err):
        return err*self.mask, None

    # Update weights
    def init_optimizer(self, optimizer, lr):
        return  # No optimizer to init for dropout layers

    # Update weights
    def update(self, grads):
        return  # No weights to update for dropout layers

    # Resample mask
    def resample(self):
        self.mask = (np.random.uniform(size=self.n_in) > self.rate).astype('float')
