import math

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
    
    # Get index of label output node
    loss_i = int(Y == 0)
    loss = -math.log(h2[loss_i])

    
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
        dk2[i] = h2[i] - int(loss_i == i)
    
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
    
    print("dW = " + str(dW))
    print("db = " + str(db))

    print("dV = " + str(dV))
    print("dc = " + str(dc))



        




X = [1., -1.]
Y = [1,0]

# weights
W = [[1.,1.,1.], [-1., -1., -1.]]
b = [0., 0., 0.]
V = [[1., 1.], [-1., -1.], [-1., -1.]]
c = [0., 0.]

linear_forward_backward(X, Y, W, b, V, c)
