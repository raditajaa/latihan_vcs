import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def relu(x):
    return np.maximum(0, x)

#INPUT 3
X = np.random.randn(3,1)
W1 = np.random.randn(4,3)
b1 = np.random.randn(4,1)

#HIDDEN LAYERS 1
Z1 = W1 @ X + b1
H1 = relu(Z1)
print(H1)

W2 = np.random.randn(4,4)
b2 = np.random.randn(4,1)

#HIDDEN LAYERS 2
Z2 = W2 @ H1 + b2
H2 = relu(Z2)
print(H2)

#OUTPUT 
W3 = np.random.randn(2,4)
b3 = np.random.randn(2,1)

Z3 = W3 @ H2 + b3
OUTPUT = sigmoid(Z2)
print(OUTPUT)