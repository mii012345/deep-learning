import numpy as np
from files import mnist
from files import neural_three_layer as neural
import os.path as p
import pickle

(x_train,t_train),(x_test,t_test) = mnist.load_mnist(normalize=True,one_hot_label=True)
network = neural.Neural(input_size=784,hidden_size=100,output_size=10)

if p.exists("params0.pkl") == True:
    network.load_params("params0.pkl")

iters_num = 100 #回数を指定します
train_size = x_test.shape[0]
batch_size = 100 #一度に学習するデータの数を指定します
learning_rate = 0.1 #学習率

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_test[batch_mask]
    t_batch = t_test[batch_mask]

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
print(a*100,"%")

network.save_params("params0.pkl")
