import cupy as np

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
    シグモイド関数の勾配
    """
    return (1.0 - sigmoid(x)) * sigmoid(x)

def cross_entropy_error(y, t):
    """cross_entropy_error
    交差エントロピー誤差
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
