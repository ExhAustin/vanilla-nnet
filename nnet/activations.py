# Activation Functions for Network Layers

import numpy as np

function_list = ['linear', 'sigmoid', 'tanh', 'relu', 'softmax']

def sigmoid:
    def __init__(self):
        # Use memory of last output to save time in backpropagation
        self.x_prev = 0
        self.o_prev = 0

    def forward(self, x):
        self.x_prev = x
        self.o_prev = 1/(1+np.exp(-x))
        return self.o_prev

    def gradient(self, err, x, repeat=False):
        if repeat:  # Use output from memory to save time
            o = self.o_prev
        else:
            o = self.forward(x)
        return err * o * (1-o)

def softmax:
    def __init__(self):
        # Use memory of last output to save time in backpropagation
        self.x_prev = 0
        self.o_prev = 0

    def forward(self, x):
        self.x_prev = x
        exp_x = np.exp(x)
        self.o_prev = exp_x / np.sum(exp_x, axis=1).reshape(-1,1)
        return self.o_prev

    def gradient(self, err, x, repeat=False):
        if repeat:  # Use output from memory to save time
            o = self.o_prev
        else:
            o = self.forward(x)
        o1 = o[...,None]
        diag_comp = np.eye(o.shape[1])[None,...] * o1
        dot_comp = np.matmul(o1, o[:,None,:])
        return np.matmul((diag_comp + dot_comp), err[...,None])[...,0]

