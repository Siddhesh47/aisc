import numpy as np

np.random.seed(1)

# Training data for XOR gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

# Neural network architecture parameters
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))
lr = 0.5

# Training loop (Backpropagation)
for epoch in range(1, 10001):
    # Forward pass
    hidden = sigmoid(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    
    # Compute error
    error = y - output
    
    # Backpropagation pass
    d_output = error * sigmoid_derivative(output)
    d_hidden = (d_output @ W2.T) * sigmoid_derivative(hidden)
    
    # Gradient update
    W2 += lr * hidden.T @ d_output
    b2 += lr * d_output.sum(axis=0, keepdims=True)
    W1 += lr * X.T @ d_hidden
    b1 += lr * d_hidden.sum(axis=0, keepdims=True)
    
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Mean Squared Error = {np.mean(error ** 2):.6f}")

# Results
print("\nFinal predictions:")
for xi, out in zip(X, output):
    print(f"Input {xi} -> {out[0]:.3f} -> Class {int(out[0] > 0.5)}")