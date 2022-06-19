""" Source Code: https://github.com/hsjeong5/MNIST-for-Numpy.git """

import gzip
import pickle
import numpy as np
from urllib import request

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images"    , "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels"    , "t10k-labels-idx1-ubyte.gz"]
]

def load_examples(size):
    train_examples, train_labels, test_examples, test_labels = get_mnist()
    training_data = [(reshape_input(x, size), vectorized_result(y)) for x, y in zip(train_examples, train_labels)]
    test_data     = [(reshape_input(x, size), y) for x, y in zip(test_examples, test_labels)]
    return (training_data, test_data)

def reshape_input(x, size):
    return np.reshape(x, (size, 1))

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def get_mnist():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"

    for name in filename:
        print(f"Downloading {name[1]} ...")
        request.urlretrieve(base_url+name[1], name[1])

    print("Download complete.")

def save_mnist():
    mnist = {}

    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)

    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)

    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

if __name__ == '__main__':
    init()
