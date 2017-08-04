import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
from . import function as f

class Neural:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b3'] = np.zeros(output_size)

    def predict(self,x):
        W1,W2,W3 = self.params['W1'],self.params['W2'],self.params['W3']
        b1,b2,b3 = self.params['b1'],self.params['b2'],self.params['b3']

        a1 = np.dot(x,W1) + b1
        z1 = f.sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        z2 = f.sigmoid(a2)
        a3 = np.dot(z2,W3) + b3
        y = f.softmax(a3)

        return y

    def gradient(self,x,t):
        W1,W2,W3 = self.params['W1'],self.params['W2'],self.params['W3']
        b1,b2,b3 = self.params['b1'],self.params['b2'],self.params['b3']
        grads = {}

        batch_num = x.shape[0]

        a1 = np.dot(x,W1) + b1
        z1 = f.sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        z2 = f.sigmoid(a2)
        a3 = np.dot(z2,W3) + b3
        y = f.softmax(a3)

        dy = (y - t) / batch_num
        grads['W3'] = np.dot(z2.T,dy)
        grads['b3'] = np.sum(dy,axis=0)

        da1 = np.dot(dy,W3.T)
        dz1 = f.sigmoid_grad(a2) * da1
        grads['W2'] = np.dot(z1.T,dz1)
        grads['b2'] = np.sum(dz1,axis=0)

        da2 = np.dot(dz1,W2.T)
        dz2 = f.sigmoid_grad(a1) * da2
        grads['W1'] = np.dot(x.T,dz2)
        grads['b1'] = np.sum(dz2,axis=0)

        return grads

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name,'rb') as f:
            obj = pickle.load(f)
            for i in range(2):
                self.params['W'+str(i+1)] = obj['W'+str(i+1)]
                self.params['b'+str(i+1)] = obj['b'+str(i+1)]
