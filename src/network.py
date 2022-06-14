import random
import numpy as np

class Network:

    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
        """ Create the pairs (Weights,Nodes) for each layer except the last one.
            Create matrices with dimension (Weights,Nodes) and randomize its values.
            -> The lines represent the weights of a node.
            -> The columns are the values of each weight.
        """
