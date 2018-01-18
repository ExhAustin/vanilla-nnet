# This example implements the simple neural network on the iris dataset

import numpy as np
from sklearn import datasets

from nnet.models import NeuralNet
from nnet.layers import Dense, Dropout

# Main function
def main():
    # Training parameters
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 10
    BATCHSIZE = 10
    VALIDATION_RATE = 0.05

    # Define neural network architecture
    net = NeuralNet(lr=LEARNING_RATE, optimizer='sgd')
    net.add_layer(Dense(n_in=4, n_out=32, activation='sigmoid'))
    #net.add_layer(Dropout(n_in=32, rate=0.2))
    #net.add_layer(Dense(n_in=32, n_out=32, activation='sigmoid'))
    #net.add_layer(Dropout(n_in=32, rate=0.2))
    net.add_layer(Dense(n_in=32, n_out=3, activation='softmax'))

    # Parse data
    parser = IrisParser()
    iris = datasets.load_iris()
    Y_train = parser.parseY(iris.target)
    X_train = parser.parseX(iris.data)
    np.random.shuffle(X_train)    # shuffle data to prevent biased validation set


    # Train
    net.train(X_train, Y_train, 
            max_epochs = MAX_EPOCHS, 
            batchsize = BATCHSIZE,
            validation_rate = VALIDATION_RATE, 
            early_stop=False,
            verbose=True)

class IrisParser:
    # Data parser
    def parseX(self, data):
        # Update normalization parameters
        self.mu_x = np.mean(data, axis=0)
        self.sigma_x = np.std(data, axis=0)

        # Normalize
        X = self.normalize(data)

        return X

    # Label parser
    def parseY(self, target):
        Y = np.zeros([len(target), np.max(target)+1])
        for i in range(len(target)):
            Y[i, target[i]] = 1
        return Y

    # Normalize data
    def normalize(self, data):
        X = (data - self.mu_x) / self.sigma_x
        return X


if __name__ == '__main__':
    main()
