import numpy as np

# Training data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Model parameters
weights = np.zeros(2)
bias = 0.0
lr = 0.1

def step(x):
    return 1 if x >= 0 else 0

# Training loop
for epoch in range(1, 21):
    errors = 0
    for xi, target in zip(X, y):
        output = step(np.dot(weights, xi) + bias)
        error = target - output
        weights = weights + lr * error * xi
        bias = bias + lr * error
        errors += abs(error)
    print(f"Epoch {epoch:2d} | weights = {np.round(weights, 2)} | bias = {bias:.2f} | errors = {errors}")
    if errors == 0:
        print("Training complete!\n")
        break

# Model testing
print("Testing the trained perceptron (AND gate):")
for xi in X:
    out = step(np.dot(weights, xi) + bias)
    print(f"Input {xi} -> Output {out}")