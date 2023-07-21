import numpy as np
import matplotlib.pyplot as plt

x = np.array([4, 3, 0])
coefficients = [np.array([-.5, .1, .08]),
                np.array([-.2, .2, .31]),
                np.array([.5, -.1, 2.53])]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 6))

colors = ['r', 'g', 'b']

# Plot sigmoid curve
weighted_sums = np.linspace(-5, 5, 100)  # Range of weighted sums for plotting
s = sigmoid(weighted_sums)
plt.plot(weighted_sums, s, color='k')

# Plot vertical lines for weighted sums
for i, c in enumerate(coefficients):
    weighted_sum = np.dot(x, c)
    plt.axvline(x=weighted_sum, color=colors[i], linestyle='--', label=f'Weighted Sum (c{i+1})')

plt.xlabel('weighted sum')
plt.ylabel('Sigmoid Output')
plt.legend()
plt.grid(True)
plt.show()
