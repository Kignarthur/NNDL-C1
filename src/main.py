import mnist
import numpy as np
import network

def main():
    layers = [784, 30, 10]
    net = network.Network(layers)

    training_data, test_data = mnist.load_examples(layers[0])
    epc = 10; mbs = 500; eta = 5.0

    net.stochastic_gradient_descent(training_data=training_data, epochs=epc,
                                    mini_batch_size=mbs, learning_rate=eta,
                                    test_data=test_data)

main()
