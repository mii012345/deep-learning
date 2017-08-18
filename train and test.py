import cupy as cp
import numpy as np
from files import neural_layer as neural
from files import tenki_num as t
import chainer.cuda as cuda
import time
import gc
import matplotlib.pyplot as plt

def testNeural(network,x_test,t_test):
    print("Start Test")
    ans = 0
    for i in np.arange(0,200,5):
        a = [i]
        x_batch = x_test[a]
        t_batch = t_test[a]
        print(i,end=" ")
        for ii in range(1,5):
            print(i+ii,end=" ")
            x_batch = np.append(x_batch,x_test[[i+ii]],axis=0)
            t_batch = np.append(t_batch,t_test[[i+ii]],axis=0)
        x_batch = cp.array(x_batch)
        t_batch = cp.array(t_batch)

        acc = network.accuracy(x_batch,t_batch)
        #network.update(x_batch,t_batch)
        print(acc)
        if acc == 1.0:
            ans += 5
        #if acc == 0.5:
            #ans += 1
        if acc == 0.8:
            ans += 4
        if acc == 0.6:
            ans += 3
        if acc == 0.4:
            ans += 2
        if acc == 0.2:
            ans += 1

    print()
    print(ans,end=" ")
    ansper = (ans/200)*100
    print(ansper)
    return ansper

print("Now Loading...")
x_train,t_train,x_test,t_test,x_acc,t_acc = t.loadPic()

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_acc = x_acc.astype(np.float32)

acc_list = []
ar = 2

network = neural.NeuralNet()
train_num = int(input("train size"))
boo = bool(input("load params?"))
if boo:
    network.load_params("params.pkl")
A = time.time()

ans = 0
print("Start Train")
for i in range(0,train_num,ar):
    #x = random.randint(0,699)
    #a = [x]
    #a = [i]

    batch_mask = np.random.choice(x_train.shape[0]-1,ar)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #x_batch = x_train[a]
    #t_batch = t_train[a]
    #for ii in range(1,2):
        #x_batch = np.append(x_batch,x_train[[i+ii]],axis=0)
        #t_batch = np.append(t_batch,t_train[[i+ii]],axis=0)
    x_batch = cp.array(x_batch)
    t_batch = cp.array(t_batch)
    network.update(x_batch,t_batch)
    del x_batch,t_batch
    gc.collect()
    print(i)
    if i>0 and i%100 == 0:
        ansper = testNeural(network,x_test,t_test)
        acc_list.append(ansper)
print("Start Test")
ans = 0
for i in np.arange(0,300,5):
    a = [i]
    x_batch = x_acc[a]
    t_batch = t_acc[a]
    print(i,end=" ")
    for ii in range(1,5):
        print(i+ii,end=" ")
        x_batch = np.append(x_batch,x_acc[[i+ii]],axis=0)
        t_batch = np.append(t_batch,t_acc[[i+ii]],axis=0)
    x_batch = cp.array(x_batch)
    t_batch = cp.array(t_batch)

    acc = network.accuracy(x_batch,t_batch)
    #network.update(x_batch,t_batch)
    print(acc)
    if acc == 1.0:
        ans += 5
    if acc == 0.8:
        ans += 4
    if acc == 0.6:
        ans += 3
    if acc == 0.4:
        ans += 2
    if acc == 0.2:
        ans += 1

print()
print(ans,end=" ")
ansper = (ans/300)*100
print(ansper)
acc_list.append(ansper)


B = time.time()
tim = int(np.round((B-A)/60))
print(tim,"min")
boo = bool(input("save params?"))
if boo:
    network.save_params("params_gpu.pkl")

x = np.arange(100,train_num+1,100)
plt.plot(x, acc_list,marker='o')
plt.xlabel("many")
plt.ylabel("accuracy")
plt.title(str(tim)+"min")
plt.ylim(0, 100.0)
plt.show()

#300,200->42.0
#350,200->50.5
#400,200->50.5
#450,200->50.5
#500,200->50.5
#550,200->39.0
#600,200->38.5


#700,200->50.5

#Momentum&hidden=50
#650,200->38.5
#700,200->38.5
