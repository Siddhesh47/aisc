import numpy as np

# Membership Function
def triangular(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

# Temperature Membership Functions
def temp_cold(t):
    return triangular(t, -20, 0, 20)

def temp_warm(t):
    return triangular(t, 10, 25, 40)

def temp_hot(t):
    return triangular(t, 30, 50, 70)

# Fan Speed Universe & Membership Functions
speed = np.arange(0, 101, 1)
slow = triangular(speed, -50, 0, 50)
medium = triangular(speed, 25, 50, 75)
fast = triangular(speed, 50, 100, 150)

# Inference Engine & Defuzzification (Centroid Method)
def fan_controller(t):
    c, w, h = temp_cold(t), temp_warm(t), temp_hot(t)
    clipped_slow = np.minimum(c, slow)
    clipped_medium = np.minimum(w, medium)
    clipped_fast = np.minimum(h, fast)
    aggregated = np.maximum(clipped_slow, np.maximum(clipped_medium, clipped_fast))
    
    if aggregated.sum() == 0:
        return 0, (c, w, h)
    return np.sum(speed * aggregated) / np.sum(aggregated), (c, w, h)

# Controller Evaluation
print("Temp(C) | Cold Warm Hot | Fan Speed (%)")
print("-" * 45)
for t in [5, 15, 25, 35, 45]:
    out, (c, w, h) = fan_controller(t)
    print(f"{t:6d} | {c:.2f} {w:.2f} {h:.2f} | {out:6.2f}")