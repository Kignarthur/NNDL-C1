import network
import mnist_loader

def main():
    layers = [784, 30, 10]
    net = network.Network(layers)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    epc = 5; mbs = 10; eta = 3.0

    net.stochastic_gradient_descent(training_data=training_data, epochs=epc,
                                    mini_batch_size=mbs, learning_rate=eta,
                                    test_data=test_data)

main()
