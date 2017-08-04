import sys,os
sys.path.append(os.pardir)
import cupy as cp
from . import layers as l
from  collections import OrderedDict

class NeuralNet:
    def __init__(self,input_dim=(3,360,360),
                 conv_param={'filter_num':30,'filter_size':5,
                             'pad':0,'stride':1},
                 hidden_size=100,output_size=3,weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size-filter_size+2*filter_pad)/filter_stride+1
        pool_output_size = int(filter_num*(conv_output_size/2)*(conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_init_std * cp.random.randn(filter_num,input_dim[0],filter_size,filter_size)
        self.params['b1'] = cp.zeros(filter_num)
        self.params['W2'] = weight_init_std * cp.random.randn(pool_output_size,hidden_size)
        self.params['b2'] = cp.zeros(hidden_size)
        self.params['W3'] = weight_init_std * cp.random.randn(hidden_size,output_size)
        self.params['b3'] = cp.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = l.Convolution(self.params['W1'],self.params['b1'],conv_param['stride'],conv_param['pad'])
        self.layers['Relu1'] = l.Relu()
        self.layers['Pool1'] = l.Pooling(pool_h=2,pool_w=2,stride=2)
        self.layers['Affine1'] = l.Affine(self.params['W2'],self.params['b2'])
        self.layers['Relu2'] = l.Relu()
        self.layers['Affine2'] = l.Affine(self.params['W3'],self.params['b3'])
        self.lastLayer = l.SoftmaxWithLoss()

        self.opt = l.AdaGrad()

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
