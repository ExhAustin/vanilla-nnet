# Optimizers for Training

import numpy as np

method_list = ['sgd', 'Adam']

# Stochastic gradient descent
class sgd:
    def __init__(self, lr, reg=0):
        self.lr = lr
        self.reg=reg

    def update_weights(self, w, g):
        w = w - self.lr * g - self.reg*w
        return w

# Adam optimizer
class Adam:
    def __init__(self, lr, reg=0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.reg=reg
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize timestep and moments
        self.t = 0
        self.m = 0
        self.v = 0

    def update_weights(self, w, g):
        # Update parameters
        self.t += 1
        self.m = self.beta1*self.m + (1-self.beta1)*g
        self.v = self.beta2*self.v + (1-self.beta2)*g*g

        # Compute bias-corrected moments
        m_bc = self.m / (1 - self.beta1**self.t)
        v_bc = self.v / (1 - self.beta2**self.t)

        # Update weights
        w = w - self.lr * m_bc/(np.sqrt(v_bc) + self.epsilon) - self.reg*w

        return w
