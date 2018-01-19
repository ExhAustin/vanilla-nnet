# This example implements the simple neural network on the iris dataset

import numpy as np
from sklearn import datasets

from nnet.models import NeuralNet
from nnet.layers import Dense, Dropout

# Main function
def main():
    #--------------------------
    # Model initialization
    #--------------------------
    # Training parameters
    LEARNING_RATE = 0.005
    MAX_EPOCHS = 70
    BATCHSIZE = 5
    VALIDATION_RATE = 0.04

    # Define neural network architecture
    net = NeuralNet(lr=LEARNING_RATE, optimizer='Adam', reg=1e-5)
    net.add_layer(Dense(n_in=4, n_out=64, activation='sigmoid'))
    net.add_layer(Dropout(n_in=64, rate=0.4))
    net.add_layer(Dense(n_in=64, n_out=16, activation='sigmoid'))
    net.add_layer(Dropout(n_in=16, rate=0.2))
    net.add_layer(Dense(n_in=16, n_out=3, activation='softmax'))

    #--------------------------
    # Data preprocessing
    #--------------------------
    # Load data
    parser = IrisParser()
    iris = datasets.load_iris()
    Y = parser.parseY(iris.target)
    X = parser.parseX(iris.data)

    # Shuffle data to prevent biased validation/test set splits
    n_features = X.shape[1]
    XY = np.concatenate([X, Y], axis=1)
    np.random.shuffle(XY)    
    X = XY[:,0:n_features]
    Y = XY[:,n_features:]

    # Split data into training and testing set
    i_split = 120
    X_train = X[0:i_split,:]
    Y_train = Y[0:i_split,:]
    X_test = X[i_split:,:]
    Y_test = Y[i_split:,:]

    # Normalize data using mean and variance of training data
    parser.getNormParams(X_train)
    X_train = parser.normalize(X_train)
    X_test = parser.normalize(X_test)

    #--------------------------
    # Training
    #--------------------------
    # Check accuracy of initial model with test set
    Y_pred = net.predict(X_test)
    comp = (np.argmax(Y_test, axis=1) == np.argmax(Y_pred, axis=1))
    acc = np.sum(comp) / comp.size
    print('Test accuracy before training:', acc*100, '%')

    # Train network
    net.train(X_train, Y_train, 
            max_epochs = MAX_EPOCHS, 
            batchsize = BATCHSIZE,
            validation_rate = VALIDATION_RATE, 
            early_stop=False,
            verbose=True)

    # Check final accuracy with test set
    Y_pred = net.predict(X_test)
    comp = (np.argmax(Y_test, axis=1) == np.argmax(Y_pred, axis=1))
    acc = np.sum(comp) / comp.size
    print('Test accuracy after training:', acc*100, '%')

# Parser for the Iris dataset
class IrisParser:
    # Data parser
    def parseX(self, data):
        X = data
        return X

    # Label parser
    def parseY(self, target):
        Y = np.zeros([len(target), np.max(target)+1])
        for i in range(len(target)):
            Y[i, target[i]] = 1
        return Y

    # Get normalization parameters of data
    def getNormParams(self, data):
        self.mu = np.mean(data, axis=0)
        self.sigma = np.std(data, axis=0)

    # Normalize data
    def normalize(self, data):
        X = (data - self.mu) / self.sigma
        return X


if __name__ == '__main__':
    main()
