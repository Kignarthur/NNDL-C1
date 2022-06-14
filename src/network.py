import random
import numpy as np
import sigmoid

class Network:

    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for y,x in zip(sizes[1:], sizes[:-1])]
        """ Create the pairs (Weights,Nodes) for each layer except the last one.
            Create matrices with dimension [Weights x Nodes] and randomize its values.
            Each column of the matrix has all the weights of a single node.
        """

    def feedforward(self, x):
        """ The activation value of a single node in the next layer is equal to σ(Σ wk*xk + bj).
            Therefore to obtain the vector of activations we multiply the
            vector a by the weight matrix W, and add the vector b of biases.
        """
        for W,b in zip(self.weights, self.bias):
            x = sigmoid.sigmoid(np.dot(W,x) + b)
        return x
