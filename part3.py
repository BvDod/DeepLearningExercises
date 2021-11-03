import math
from data import load_synth
import random
import matplotlib.pyplot as plt
import seaborn

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def softmax(x_i, x_list):
    return math.exp(x_i)/sum([math.exp(x) for x in x_list])


def linear_forward_backward(X, Y, W, b, V, c,):

    k1 = [0., 0., 0.]
    h1 = [0., 0., 0.]

    k2 = [0., 0.]
    h2 = [0., 0.]

    # Linear forward 1: k1 = Wx + b
    for i in range(len(X)):
        for j in range(len(W[i])):
            k1[j] += X[i] * W[i][j]
    for i in range(len(b)):        
        k1[i] += b[i]

    # Activation 1: h1 = sigmoid(k1)
    for i in range(len(k1)):
        h1[i] =  sigmoid(k1[i])

    # linear forward 2: k2 = Vk1 + c
    for i in range(len(h1)):
        for j in range(len(V[i])):
            k2[j] += h1[i] * V[i][j]
    for i in range(len(c)):        
        k1[i] += c[i]

    # Activation 2: h2 = softmax(k2)
    for i in range(len(k2)):
        h2[i] =  softmax(k2[i], k2)
    
    # calc loss
    loss = -math.log(h2[Y])

    
    # Gradients
    dk1 = [0., 0., 0.]
    dh1 = [0., 0., 0.]
    dk2 = [0., 0.]
    dh2 = [0., 0.]
    dW = [[0., 0., 0.],[0., 0., 0.]]
    db = [0., 0., 0.]
    dV = [[0., 0.,],[0., 0.,],[0., 0.,]]
    dc = [0., 0.,]

    # dloss/dk2
    for i in range(len(k2)):
        dk2[i] = h2[i] - int(Y == i)
    
    # dloss/dV
    for j in range(len(V)):
        for i in range(len(V[i])):
            dV[j][i] = h1[j] * dk2[i]
    
    # dloss/dc
    for i in range(len(c)):
        dc[i] = dk2[i]
    
    # dloss/dh1
    for i in range(len(k1)):
        for j in range(len(V[i])):
            dh1[i] += V[i][j] * dk2[j]
    # dloss/dk1
    for i in range(len(k1)):
        dk1[i] = dh1[i] * h1[i] * (1- h1[i]) 

    # dloss/dW
    for j in range(len(W)):
        for i in range(len(W[j])):
            dW[j][i] = X[j]*dk1[i]

    # dloss/db
    for i in range(len(b)):
        db[i] = dk1[i]

    return dW, db, dV, dc, loss

(xtrain, ytrain), (xval, yval), num_cls = load_synth()

print(len(xtrain))
print(ytrain)
W = [[random.gauss(0,1), random.gauss(0,1), random.gauss(0,1)], [random.gauss(0,1),random.gauss(0,1),random.gauss(0,1)]]
b = [0., 0., 0.]
V = [[random.gauss(0,1), random.gauss(0,1)], [random.gauss(0,1), random.gauss(0,1)], [random.gauss(0,1), random.gauss(0,1)]]
c = [0., 0.]

alpha = 0.025
losses_mean = []
steps_at_losses = []
steps = 100000
for step in range(steps):
    
    i = random.randrange(0, len(xtrain))
    dW, db, dV, dc, loss = linear_forward_backward(xtrain[i], ytrain[i], W, b, V, c)

    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] -= alpha * dW[i][j]

    for i in range(len(b)):
        b[i] -= alpha * db[i]

    for i in range(len(V)):
        for j in range(len(V[i])):
            V[i][j] -= alpha * dV[i][j]
    
    for i in range(len(c)):
        c[i] -= alpha * dc[i]

    if ((step % 1000) == 0):
        losses = []
        for i in range(5000):
            i = random.randrange(0, len(xtrain))
            dW, db, dV, dc, loss = linear_forward_backward(xtrain[i], ytrain[i], W, b, V, c)
            losses.append(loss)
        losses_mean.append(sum(losses)/len(losses))
        steps_at_losses.append(step)

seaborn.set()
plt.plot(steps_at_losses, losses_mean)
plt.title("Training loss")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
