import cupy as cp
from files import mnist
from files import neural_layer as neural
from files import tenki_num as t
import matplotlib.pyplot as plt
import gc
import chainer.cuda as cuda
import time

print("Now Loading...")
mas,tenki,mastest,tenkitest = t.loadPic()
mas = mas.astype(cp.float32)
mas /= 255.0
x_train = mas[:350]
t_train = tenki[:350]
x_test = mastest[:200]
t_test = tenkitest[:200]
A = time.time()

del mas,tenki,mastest,tenkitest
gc.collect()

network = neural.NeuralNet()
ans = 0
print("Start Train")
for i in range(350):
    a = [i]
    x_batch = x_train[a]
    t_batch = t_train[a]
    grad = network.gradient(x_batch,t_batch)
    for key in ('W1','b1','W2','b2','W3','b3'):
        network.params[key] -= 0.01 * grad[key]
    #network.update(x_batch,t_batch)

    y = network.predict(x_batch)
    if cp.argmax(y[0,:]) == cp.argmax(t_batch[0,:]):
        answer = "〇"
    else:
        answer = "×"
    print(i,"seikai",cp.argmax(t_batch[0,:]),"yosoku",cp.argmax(y[0,:]),answer)
B = time.time()
print(B-A)
#for key in ('W1','b1','W2','b2','W3','b3'):
    #network.params[key] = cuda.to_cpu(network.params[key])
    #print(type(network.params[key]))
print("Start Test")
for i in range(200):
    a = [i]
    x_batch = x_test[a]
    t_batch = t_test[a]
    #network.update(x_batch,t_batch)

    y = network.predict(x_batch)
    if cp.argmax(y[0,:]) == cp.argmax(t_batch[0,:]):
        answer = "〇"
        ans += 1
    else:
        answer = "×"
    print(i,"seikai",cp.argmax(t_batch[0,:]),"yosoku",cp.argmax(y[0,:]),answer)
print((ans/200)*100)
#300,200->47.5
