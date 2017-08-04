# coding: utf-8
import cupy as cp
from files.function import *
from files.util import im2col, col2im
import gc

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        #print("relu")
        self.mask = (x <= 0)
        out = x.copy()
        del x
        gc.collect()
        out[self.mask] = 0
        #print(" out")

        return out
        del out
        gc.collect()

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
        del dout,dx
        gc.collect()

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        #print("affine")
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        #print(" x")
        del x
        gc.collect()
        out = self.x.dot(self.W) + self.b
        #print(" out")

        return out
        del out
        gc.collect()

    def backward(self, dout):
        dx = dout.dot(self.W.T)
        self.dW = self.x.T.dot(dout)
        self.db = cp.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx
        del dx
        gc.collect()

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = cp.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = cp.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (cp.sqrt(self.h[key]) + 1e-7)


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
        del self.loss
        gc.collect()

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[cp.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
        del batch_size,dx
        gc.collect()

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

        del W,b,stride,pad
        gc.collect()

    def forward(self, x):
        #print("conv")
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        #print(" out_h,out_w")

        col = im2col(x, FH, FW, self.stride, self.pad)
        #print(" im2col")
        col_W = self.W.reshape(FN, -1).T
        #print(" col_W")
        out = col.dot(col_W) + self.b
        #out = np.dot(col,col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        #print(" out")

        self.x = x
        self.col = col
        self.col_W = col_W
        del col,col_W,x,out_h,out_w,FN,C,FH,FW,N,H,W
        gc.collect()

        return out


    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = cp.sum(dout, axis=0)
        #self.dW = np.dot(self.col.T, dout)
        self.dW = self.col.T.dot(dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        #dcol = np.dot(dout, self.col_W.T)
        dcol = dout.dot(self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
        del FN,C,FH,FW,dout,dcol,dx
        gc.collect()


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

        del pool_h,pool_w,stride,pad
        gc.collect()

    def forward(self, x):
        #print("pool")
        N, C, H, W = x.shape
        #print(" x.shape")
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        #print(" out_h,out_w")

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        #print(" im2col")
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = cp.argmax(col, axis=1)
        out = cp.max(col, axis=1)
        del col
        gc.collect()
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out
        del x,arg_max,out,N,C,out_h,out_w
        gc.collect()

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = cp.zeros((dout.size, pool_size))
        dmax[cp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
        del dout,pool_size,dmax,dcol,dx
        gc.collect()
