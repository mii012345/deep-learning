import sys,os
sys.path.append(os.pardir)
import cupy as cp
import numpy as np
from . import layers as l
from  collections import OrderedDict
import gc
import pickle

class NeuralNet:
    def __init__(self,input_dim=(3,360,360),
                 conv_param={'filter_num':30,'filter_size':5,
                             'pad':0,'stride':1},
                 hidden_size=50,output_size=3,weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size-filter_size+2*filter_pad)/filter_stride+1
        pool_output_size = int(filter_num*(conv_output_size/2)*(conv_output_size/2))
        self.lr =0.1

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num,input_dim[0],filter_size,filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size,hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        params_gpu = {}
        for key in ['W1','b1','W2','b2','W3','b3']:
            params_gpu[key] = cp.array(self.params[key])
        self.layers['Conv1'] = l.Convolution(params_gpu['W1'],params_gpu['b1'],conv_param['stride'],conv_param['pad'])
        self.layers['Relu1'] = l.Relu()
        self.layers['Pool1'] = l.Pooling(pool_h=2,pool_w=2,stride=2)
        self.layers['Affine1'] = l.Affine(params_gpu['W2'],params_gpu['b2'])
        self.layers['Relu2'] = l.Relu()
        self.layers['Affine2'] = l.Affine(params_gpu['W3'],params_gpu['b3'])
        self.layers['Dropout'] = l.Dropout()

        self.lastLayer = l.SoftmaxWithLoss()

        self.opt = l.AdaGrad(lr=self.lr)

        del params_gpu
        gc.collect()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = cp.argmax(y,axis=1)
        if t.ndim != 1 : t = cp.argmax(t,axis=1)

        accuracy = cp.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self,x,t):
        self.loss(x,t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def update(self,x,t):
        grad = self.gradient(x,t)
        self.opt.update(self.params,grad)

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = cp.array(self.params['W' + str(i+1)])
            self.layers[key].b = cp.array(self.params['b' + str(i+1)])
