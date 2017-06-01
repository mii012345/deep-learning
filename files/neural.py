import numpy as np
from . import function as f

class Neural:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = f.sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = f.softmax(a2)

        return y

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
        dz1 = f.sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T,dz1)
        grads['b1'] = np.sum(dz1,axis=0)

        return grads

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
