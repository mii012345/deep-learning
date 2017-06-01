import numpy as np
from files import mnist
from files import neural

(x_train,t_train),(x_test,t_test) = mnist.load_mnist(normalize=True,one_hot_label=True)

network = neural.Neural(input_size=784,hidden_size=100,output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch,t_batch)

    #print(grad)

    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]

    y = network.predict(x_batch)
    print("seikai",np.argmax(t_batch[0,:]),"yosoku",np.argmax(y[0,:]))
    a=network.accuracy(x_train,t_train)
    print(a)
