import numpy as np

def sigmoid(x):
    """sigmoid
    シグモイド関数
    """
    y = 1 / (1 + np.exp(-x))
    return y

def softmax(x):
    """softmax
    ソフトマックス関数
    """
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)

    return exp_x / sum_exp_x
