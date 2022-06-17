import numpy as np

def σ(z):
    """ Sigmoid function"""
    return 1/(1 + np.exp(-z))

def σp(z):
    """ Sigmoid first derivative"""
    return σ(z) * (1-σ(z))

