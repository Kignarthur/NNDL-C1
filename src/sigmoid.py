import warnings
from numpy import exp

warnings.filterwarnings('ignore')

def σ(z):
    """ Sigmoid function"""
    return 1.0/(1.0 + exp(-z))

def σp(z):
    """ Sigmoid first derivative"""
    return σ(z) * (1 - σ(z))
