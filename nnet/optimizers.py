# Optimizers for Training

import numpy as np

method_list = ['sgd', 'Adam']

def sgd:
    def __init__(self, lr):
        self.lr = lr

    def update_weights(self, weight_list, grad_list):
        for w, g in zip(weight_list, grad_list):
            w += self.lr * g

def Adam:
    def __init__(self, lr, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update_weights(self, weight_list, grad_list):
        return
