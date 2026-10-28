import numpy as np

# Fundamental patterns to store
patterns = np.array([
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, 1, 1, 1, -1, -1, -1, -1],
])

n = patterns.shape[1]
W = np.zeros((n, n))

# Hebbian learning for weight matrix
for p in patterns:
    W += np.outer(p, p)
np.fill_diagonal(W, 0)

print("Weight matrix:\n", W.astype(int))

def recall(pattern, max_iter=10):
    state = pattern.copy()
    for _ in range(max_iter):
        old = state.copy()
        for i in range(n):
            net = np.dot(W[i], state)
            state[i] = 1 if net >= 0 else -1
        if np.array_equal(state, old):
            break
    return state

# Test recall with noisy input
noisy = patterns[0].copy()
noisy[1] *= -1
noisy[4] *= -1

print("\nStored pattern :", patterns[0])
print("Noisy input    :", noisy)
result = recall(noisy)
print("Recalled output:", result)
print("Recall successful?", np.array_equal(result, patterns[0]))