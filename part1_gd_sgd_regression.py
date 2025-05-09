
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading data
data = pd.read_csv("Part1_x_y_Values.txt", sep='\s+')
X = data["X"].values
Y = data["Y"].values

# Create polynomial features
def create_poly_features(X, degree=2, normalize=True):
    poly = np.vstack([X**i for i in range(degree, -1, -1)]).T
    if normalize:
        mean = poly.mean(axis=0)
        std = poly.std(axis=0)
        poly = (poly - mean) / (std + 1e-8)
    return poly

def gradient_descent(X, y, lr=0.01, epochs=100):
    m, n = X.shape
    weights = np.zeros(n)
    history = []

    for epoch in range(epochs):
        predictions = X @ weights
        errors = predictions - y
        gradients = (2/m) * X.T @ errors
        weights -= lr * gradients
        history.append(weights.copy())
        if epoch % 10 == 0:
            plt.plot(X[:, 1], predictions, label=f"Epoch {epoch}")

    return weights, history

# Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, lr=0.01, epochs=100):
    np.random.seed(42)
    m, n = X.shape
    weights = np.zeros(n)
    history = []

    for epoch in range(epochs):
        for i in range(m):
            xi = X[i].reshape(1, -1)
            yi = y[i]
            prediction = xi @ weights
            error = prediction - yi
            gradient = 2 * xi.T * error
            weights -= lr * gradient.flatten()
        history.append(weights.copy())
        if epoch % 10 == 0:
            plt.plot(X[:, 1], X @ weights, label=f"Epoch {epoch}")

    return weights, history

# Preparing features to be displayed
X_poly = create_poly_features(X, degree=2)

plt.figure(figsize=(10, 5))
weights_gd, history_gd = gradient_descent(X_poly, Y, lr=0.01, epochs=100)
plt.title("Gradient Descent Fitting Curves")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.savefig("gd_fitting_curve.png")
plt.close()

# Using plotting 
plt.figure(figsize=(10, 5))
weights_sgd, history_sgd = stochastic_gradient_descent(X_poly, Y, lr=0.01, epochs=100)
plt.title("Stochastic Gradient Descent Fitting Curves")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.savefig("sgd_fitting_curve.png")
plt.close()

# Printing the final weights
print("Final GD Weights:", weights_gd)
print("Final SGD Weights:", weights_sgd)
