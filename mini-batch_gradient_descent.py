import numpy as np
from LinearRegression import gradientCalculate

# Define dataset and model parameters
X = np.random.rand(1000, 2)
y = np.random.rand(1000)
learning_rate = 0.01
epochs = 1000
batch_size = 10

# Initialize model parameters
theta = np.random.rand(3)  # Initialize with random values

# Mini-batch gradient descent
for epoch in range(epochs):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    for i in range(0, len(X), batch_size):
        # Select a mini-batch of data
        X_mini_batch = X[i:i + batch_size]
        y_mini_batch = y[i:i + batch_size]

        # Compute the gradient for the mini-batch
        gradient, cost = gradientCalculate(X, y, theta)

        # Update model parameters
        theta -= learning_rate * gradient

    # Calculate the cost for monitoring convergence
    print(f"Epoch {epoch + 1}/{epochs}, Cost: {cost}")

print("Optimized theta:", theta)
