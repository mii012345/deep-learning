import numpy as np

def sigmoid(x):
    """sigmoid
    シグモイド関数
    """
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """softmax
    ソフトマックス関数
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid_grad(x):
    """sigmoid_grad
    シグモイド関数の勾配"""
    return (1.0 - sigmoid(x)) * sigmoid(x)
