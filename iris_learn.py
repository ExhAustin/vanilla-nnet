# This example impplements the simple neural network in this project

import numpy
from sklearn import datasets

from nnet.models import NeuralNet
from nnet.layers import Dense, Dropout

# Main function
def main():
    # Training parameters
    lr = 0.1
    validation_rate = 0.1
    max_epochs = 10
    batchsize = 10

    # Define neural network architecture
    net = NeuralNet()
    net.add_layer(Dense(n_in=4, n_out=20, activation='sigmoid'))
    #net.add_layer(Dropout(n_in=20, rate=0.2))
    #net.add_layer(Dense(n_in=20, n_out=20, activation='sigmoid'))
    #net.add_layer(Dropout(n_in=20, rate=0.2))
    net.add_layer(Dense(n_in=20, n_out=3, activation='softmax'))

    # Parse data
    parser = IrisParser()
    X_train = parser.parseX

    # Train
    net.train(X_train, Y_train, optimizer='Adam', verbose=True)

    # Validate
    Y_predict = net.predict(X_validate, verbose=True)


def IrisParser:
    def 


if __name__ == '__main__':
