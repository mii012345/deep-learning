import numpy as np
import function as f

def sigmoid_grad(x):
    return (1.0 - f.sigmoid(x)) * f.sigmoid(x)

def gradient(self,x,t):
    W1,W2 = self.params['W1'],self.params['W2']
    b1,b2 = self.params['b1'],self.params['b2']
    grads = {}

    batch_num = x.shape[0]

    a1 = np.dot(x,W1) + b1
    z1 = f.sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    y = f.softmax(a2)

    dy = (y - t) / batch_num
    grads['W2'] = np.dot(z1.T,dy)
    grads['b2'] = np.sum(dy,axis=0)

    da1 = np.dot(dy,W2.T)
    dz1 = self.sigmoid_grad(a1) * da1
    grads['W1'] = np.dot(x.T,dz1)
    grads['b1'] = np.sum(dz1,axis=0)

    return grads
