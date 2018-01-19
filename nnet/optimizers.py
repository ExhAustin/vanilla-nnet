# Optimizers for Training

import numpy as np

method_list = ['sgd', 'Adam']

# Stochastic gradient descent
class sgd:
    def __init__(self, lr):
        self.lr = lr

    def update_weights(self, w, g):
        w = w - self.lr * g
        return w

# Adam optimizer
class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0
        self.m = 0
        self.v = 0

    def update_weights(self, w, g):
        # Update parameters
        self.t += 1
        self.lr = self.lr * np.sqrt(1-self.beta2**self.t) / (1-self.beta1**self.t)
        self.m = self.beta1*self.m + (1-self.beta1)*g
        self.v = self.beta2*self.v + (1-self.beta2)*g*g
        print(self.m)

        # Update weights
        w = w - self.lr * self.m/(np.sqrt(self.v) + self.epsilon)

        return w
