import numpy as np

np.random.seed(0)

# Input data clusters
data = np.array([
    [1.0, 1.1], [1.2, 0.9], [0.8, 1.0],
    [5.0, 5.2], [5.1, 4.8], [4.9, 5.0],
    [1.0, 5.0], [1.1, 5.2], [0.9, 4.9],
])

# Network parameters
n_neurons = 3
weights = np.random.rand(n_neurons, 2) * 6
epochs = 100
lr0 = 0.5

# Training loop
for epoch in range(epochs):
    lr = lr0 * (1 - epoch / epochs)
    for x in data:
        distances = np.linalg.norm(weights - x, axis=1)
        winner = np.argmin(distances)
        for j in range(n_neurons):
            neighbour_effect = 1.0 if j == winner else (0.3 if abs(j - winner) == 1 else 0.0)
            weights[j] += lr * neighbour_effect * (x - weights[j])

# Results
print("Final neuron weights (cluster centres):")
for i, w in enumerate(weights):
    print(f"Neuron {i}: {np.round(w, 2)}")

print("\nCluster assigned to each data point:")
for x in data:
    winner = np.argmin(np.linalg.norm(weights - x, axis=1))
    print(f"Point {x} -> Neuron {winner}")