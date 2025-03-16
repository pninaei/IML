import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Create a synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Define k values to evaluate
k_values = [1, 10]

# Create a mesh grid for plotting
h = .02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot decision boundaries for each k value
plt.figure(figsize=(12, 6))
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

for i, k in enumerate(k_values):
    # Train the k-NN classifier
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)

    # Predict the class for each point in the mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.subplot(1, 2, i + 1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cm_bright)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolor='k')
    plt.title(f'k-NN (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
