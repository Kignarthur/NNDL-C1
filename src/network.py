import random
import numpy as np
from sigmoid import σ, σp

class Network:

    def __init__(self, layers: list[int]):
        """
        Sets up the network with the number of neurons in each
        layer defined by ``layers`` and initializes the weights
        and biases using a Normal Distribution N(0,1).
        """
        self.n_layers = len(layers)
        self.biases  = [np.random.randn(bias, 1) for bias in layers[1:]]
        self.weights = [np.random.randn(weights, node) for weights, node in zip(layers[1:], layers[:-1])]
        """
        Create a matrix for each layer where the columns are the nodes and
        the lines are the weights of a single node.
        """

        # Store values to backpropagate
        self.Zs = None # Signals
        self.As = None # Activations

    def evaluate_test_data(self, test_data: list[tuple]):
        """
        Returns the number of test inputs for which the
        neural network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)

    def feedforward(self, x: np.ndarray):
        """ Returns the output of the network for the input ``x``."""
        for W, b in zip(self.weights, self.biases): x = σ(np.dot(W, x) + b)
        return x

    def stochastic_gradient_descent(self, training_data: list[tuple], epochs: int,
                                    mini_batch_size: int, learning_rate: float,
                                    test_data: list[tuple] = None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        The ``training_data`` is a list of tuples ``(x, y)`` representing
        the training inputs and the labeled outputs. If ``test_data`` is
        provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.
        """
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = \
                [training_data[k:k+mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_weights_and_biases(mini_batch, learning_rate)

            if test_data:
                print(f"Epoch {epoch}: {self.evaluate_test_data(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete")

    def update_weights_and_biases(self, mini_batch: list[tuple], η: float):
        """
        Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch ``mini_batch``
        and a learning rate ``η``.
        """
        m = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Sum over all the elements in the mini-batch
        x = tuple((mini_batch[i][0] for i in range(m)))
        y = tuple((mini_batch[i][1] for i in range(m)))
        X  = np.column_stack(x)
        Y  = np.column_stack(y)
        nabla_b, nabla_w = self.backpropagate(X, Y)

        # Gradient Descent Update
        self.biases  = [b - (η/m) * nb.sum(axis=1).reshape(-1,1) for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (η/m) * nw for w, nw in zip(self.weights, nabla_w)]

    def backpropagate(self, X: np.ndarray, Y: np.ndarray):
        """
        Returns a tuple ``(nabla_b, nabla_w)`` representing the gradient
        for the cost function of m training examples where ``X`` is a
        matrix whose columns are the vectors in the mini-batch and ``Y``
        is a matrix whose columns are the labels.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        last_layer = self.n_layers - 1
        L = -1

        # Feed forward
        self.feedforward_training_matrix(X)

        # Compute δ of the last layer L
        Z_L = self.Zs[L]
        A_L = self.As[L]
        δ = self.cost_derivative(A_L, Y) * σp(Z_L)

        # Compute gradient for the cost function C_x
        nabla_b[L] = δ
        nabla_w[L] = np.dot(δ, self.As[L-1].T)

        # Backpropagate
        for l in reversed(range(1, last_layer)):
            Z = self.Zs[l-1]
            AT = self.As[l-1].T
            WT = self.weights[l].T

            δ = np.dot(WT, δ) * σp(Z)
            nabla_b[l-1] = δ
            nabla_w[l-1] = np.dot(δ, AT)

        return (nabla_b, nabla_w)

    def feedforward_training_matrix(self, A: np.ndarray):
        """
        Updates the signals and the activations arrays of the network where
        ``A`` is a matrix whose columns are the vectors in the mini-batch.
        """
        batch_size = A.shape[1]
        self.Zs = []
        self.As = [A]

        for W, b in zip(self.weights, self.biases):
            B = np.repeat(b, batch_size, axis=1) # Create a matrix of biases j x m
            Z = np.array(np.dot(W, A) + B)
            A = σ(Z)
            self.Zs.append(Z)
            self.As.append(A)

    def cost_derivative(self, network_output: np.ndarray, y: np.ndarray):
        """
        Returns the derivative of the cost function given the
        ``network_output`` and the correct label ``y``.
        """
        return (network_output - y)
