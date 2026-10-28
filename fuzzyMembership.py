import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Membership Function Definitions
def triangular(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

def trapezoidal(x, a, b, c, d):
    return np.maximum(np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)), 0)

def gaussian(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# Define Universe of Discourse
x = np.arange(0, 51, 1)
cold = trapezoidal(x, -1, 0, 10, 20)
warm = triangular(x, 15, 25, 35)
hot = trapezoidal(x, 30, 40, 50, 51)

def fuzzify(temp):
    return {
        "Cold": float(trapezoidal(temp, -1, 0, 10, 20)),
        "Warm": float(triangular(temp, 15, 25, 35)),
        "Hot": float(trapezoidal(temp, 30, 40, 50, 51)),
    }

# Test fuzzification
for t in [5, 18, 25, 32, 45]:
    result = fuzzify(t)
    print(f"Temperature = {t:2d} C -> " + ", ".join(f"{k}: {v:.2f}" for k, v in result.items()))

# Plotting
plt.figure(figsize=(7, 4))
plt.plot(x, cold, label="Cold (trapezoidal)")
plt.plot(x, warm, label="Warm (triangular)")
plt.plot(x, hot, label="Hot (trapezoidal)")
plt.plot(x, gaussian(x, 25, 5), "--", label="Gaussian (mean=25)")
plt.xlabel("Temperature (C)")
plt.ylabel("Membership degree")
plt.title("Fuzzy Membership Functions")
plt.legend()
plt.grid(True)
plt.savefig("exp9_plot.png", dpi=120, bbox_inches="tight")
print("\nGraph saved as exp9_plot.png")