# Activation Functions for Network Layers

import numpy as np

function_list = ['linear', 'sigmoid', 'tanh', 'relu', 'softmax']

def linear:
    def forward(self, x):
        return x
    def gradient(self, x):
        return 0*x + 1

def sigmoid:
    def forward(self, x):
        return 1/(1+np.exp(-x))
    def gradient(self, x):
        forward_x = self.forward(x)
        return forward_x * (1-forward_x)

def tanh:
    def forward(self, x):
        return np.tanh(x)
    def gradient(self, x):
        return 1 - np.tanh(x)**2

def relu:
    def forward(self, x):
        return (x > 0)*x
    def gradient(self, x):
        return (x > 0).astype('float')

def softmax: #TODO
    def forward(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    def gradient(self, x):
        return
