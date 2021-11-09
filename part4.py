from data import load_mnist
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def softmax(Z):
    Z = np.asarray(Z)
    Z = Z - np.max(Z, axis = 1, keepdims=True)
    e = np.exp(Z)
    e_sum = np.sum(e, axis = 1)
    a = e / e_sum[:,np.newaxis]
    return a
    
def linear_forward_backward(X, Y, W1, b1, W2, b2, perform_backwards = True):

    m = X.shape[0]

    z1 = (X @ W1) + b1
    a1 = sigmoid(z1)
    z2 = (a1 @ W2) + b2
    a2 = softmax(z2)

    loss = np.sum(-np.log(a2) * Y, axis=1)

    predictions = np.zeros_like(a2)
    predictions[np.arange(predictions.shape[0]), np.argmax(a2, axis=1)] = 1
    prediction_accuracy = np.mean(((predictions == 1) & (predictions == Y)).any(axis=1))
    
    if perform_backwards:
        dz2 = a2 - Y
        dW2 = a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * a1* (1-a1)
        dW1 = X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
    else:
        dW1, db1, dW2, db2 = 0, 0, 0, 0

    loss = np.sum(loss) * (1/m)

    return dW1, db1, dW2, db2, loss, prediction_accuracy




(xtrain, ytrain), (xtest, ytest), output_size = load_mnist()
print(ytrain)

W1, W2 = np.random.normal(size=(784,300)), np.random.normal(size=(300,10))
b1, b2 = np.zeros((1,300)), np.zeros((1,10))

Y_hot = np.zeros((len(ytrain), 10))
rows = np.arange(len(ytrain))
Y_hot[rows, ytrain] = 1

Y_test_hot = np.zeros((len(ytest), 10))
rows = np.arange(len(ytest))
Y_test_hot[rows, ytest] = 1

alpha = 0.001
batch_size = 512
losses = []
accuracies = []

val_losses = []
val_accuracies = []
steps_at_losses = []
epochs = 1

step = 0
for epoch in range(epochs):
    i_list = np.arange(xtrain.shape[0])
    np.random.shuffle(i_list)

    chunks = xtrain.shape[0] // batch_size
    i_splitted = np.array_split(i_list, chunks)
    i_splitted = i_splitted

    for i in i_splitted:
        dW1, db1, dW2, db2, loss, prediction_accuracy = linear_forward_backward(xtrain[i,:], Y_hot[i, :], W1, b1, W2, b2)
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2

        if not batch_size == 1:
            losses.append(loss)
            accuracies.append(prediction_accuracy)
            steps_at_losses.append(step)


            i = np.random.choice(range(len(xtest)), size=512)
            dW1, db1, dW2, db2, loss, prediction_accuracy = linear_forward_backward(xtest[i,:], Y_test_hot[i, :], W1, b1, W2, b2, perform_backwards=False)
            val_losses.append(loss)
            val_accuracies.append(prediction_accuracy)

        if batch_size == 1:
            if ((step % 100) == 0):
                i = np.random.choice(range(len(xtrain)), size=1000)
                dW1, db1, dW2, db2, loss, prediction_accuracy = linear_forward_backward(xtrain[i,:], Y_hot[i, :], W1, b1, W2, b2, perform_backwards=False)
                losses.append(loss)
                accuracies.append(prediction_accuracy)
                steps_at_losses.append(step)
        step += 1
    

seaborn.set()
plt.plot(steps_at_losses, losses)
plt.title("Training loss")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()

plt.plot(steps_at_losses, accuracies)
plt.title("Training accuracy")
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.show()

plt.plot(steps_at_losses, val_losses)
plt.title("Validation loss")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()

plt.plot(steps_at_losses, val_accuracies)
plt.title("Validation accuracy")
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.show()

