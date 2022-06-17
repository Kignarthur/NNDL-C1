import random
import numpy as np
from sigmoid import σ, σp

class Network:

    def __init__(self, layers):
        self.layers  = layers
        self.biases  = [np.random.randn(bias, 1) for bias in layers[1:]]
        self.weights = [np.random.randn(weights, node) for weights,node in zip(layers[1:], layers[:-1])]
        """ Create the pairs (Weights,Node) for each layer except the last one.
            Create matrices with dimension [Weights x Node] and randomize its values.
            Each column of the matrix has all the weights of a single node.
        """
        self.num_layers = len(layers)

        # Store values to backpropagate
        self.zs = []
        self.activations = []

    def feedforward(self, x):
        """ Return the output of the network for the input x."""
        for W, b in zip(self.weights, self.bias): x = σ(np.dot(W, x) + b)
        return x

    def feedforward_training_example(self, x):
        """ Return the output of the network for the input x."""
        """ The activation value of a single node in the next layer is equal to σ(Σ w_jk*x_k + b_j).
            Therefore to obtain the vector of activations we multiply the vector a_l by the weight
            matrix W, and add the vector b of biases. We then apply the function σ elementwise.
        """
        self.activations = [x]

        for W, b in zip(self.weights, self.bias):
            z = np.dot(W, x) + b
            x = σ(z)
            self.zs.append(z)
            self.activations.append(x)

        return x

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, η, test_data=None):
        N = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = \
                [training_data[k:k+mini_batch_size]
                for k in range(0, N, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_weights_and_biases(mini_batch, η)

            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print("Epoch {epoch} complete")

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        L = self.num_layers - 1

        # Feed forward
        self.feedforward(x)

        # Compute δ of the last layer L
        z_L = self.zs[L]
        a_L = self.activations[L]
        δ = self.cost_derivative(a_L, y) * σp(z_L)

        # Compute gradient for the cost function C_x
        nabla_b[L] = δ
        nabla_w[L] = np.dot(δ, self.activations[L-1].T)

        # Backpropagate
        for l in reversed(range(1, L)):
            δ = np.dot(self.weights[l].T, δ) * σp(self.zs[l])

            nabla_b[l] = δ
            nabla_w[l] = np.dot(δ, self.activations[l-1].T)

        return (nabla_b, nabla_w)

    def cost_derivative(self, network_output, y):
        return (network_output - y)
