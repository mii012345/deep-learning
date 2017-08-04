import numpy as np
from files import mnist
#from files import neural_three_layer as neural
import os.path as p
import pickle
(x_train,t_train),(x_test,t_test) = mnist.load_mnist(normalize=True,flatten=False,one_hot_label=True)
print(x_train.shape)
#network = neural.Neural(input_size=784,hidden_size=100,output_size=10)
'''
input_sizeは入力の数に合わせてください。
hidden_sizeは隠れ層のニューロンの数です。お好きな数字でどうぞ
output_sizeは出力の数です。今回は数字識別なので10です。

if p.exists("params0.pkl"):
    network.load_params("params0.pkl")

iters_num = 10000 #回数を指定します
train_size = x_train.shape[0]
batch_size = 100 #一度に学習するデータの数を指定します
learning_rate = 0.1 #学習率

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch,t_batch)

    for key in ('W1','b1','W2','b2','W3','b3'):
        network.params[key] -= learning_rate * grad[key]

    y = network.predict(x_batch)
    if np.argmax(y[0,:]) == np.argmax(t_batch[0,:]):
        answer = "〇"
    else:
        answer = "×"
    print("seikai",np.argmax(t_batch[0,:]),"yosoku",np.argmax(y[0,:]),answer)



a=network.accuracy(x_test,t_test)
print(a)
network.save_params("params0.pkl")
'''
