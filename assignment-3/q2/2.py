import numpy as np
import matplotlib.pyplot as plt

# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data = np.array(data)

        # TODO

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        u = (x - xi) / self.bandwidth
        norm_u = np.linalg.norm(u)
        if norm_u <= 1:
            return 3 / 4 * (1 - norm_u ** 2)
        return 0
        # TODO

    def evaluate(self, x):
        """Evaluate the KDE at point x."""
        n = len(self.data)
        return np.sum([self.epanechnikov_kernel(x, xi) for xi in self.data]) / (n * self.bandwidth)
        # TODO


# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# TODO: Initialize the EpanechnikovKDE class
kde = EpanechnikovKDE(bandwidth=1)



# TODO: Fit the data
kde.fit(data)


# Evaluate the KDE on the grid
x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])

x_range = np.linspace(x_min - 1, x_max + 1, 100)
y_range = np.linspace(y_min - 1, y_max + 1, 100)
X, Y = np.meshgrid(x_range, y_range)

Z = np.array([[kde.evaluate(np.array([x, y])) for x in x_range] for y in y_range])


# TODO: Plot the estimated density in a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
ax.scatter(data[:, 0], data[:, 1], np.zeros_like(data[:, 0]), color='red', s=10, label='Data Points')
ax.set_title('3D KDE of Transaction Data')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Density')
ax.legend()

# TODO: Save the plot
plt.savefig('transaction_distribution.png')
plt.show() 