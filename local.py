from LinearRegression import gradientCalculate
import numpy as np
import time

# Generate samples
X1 = 2 * np.random.rand(1000, 1)
X2 = -2 * np.random.rand(1000, 1)
X = np.concatenate((X1, X2), axis=1)
y = 4 + 3 * X1 + 2 * X2 + np.random.randn(1000, 1)

# Parameters for the regression
params = [3., 4., 7.]
learning_rate = 0.05

params = np.asarray(params)

total_time = 0

# Fit the model
for i in range(1000):
    start = time.time()
    grad, mse = gradientCalculate(X, y, params)
    # print(params, grad, mse)
    # Change gradient according to learning rate
    params -= learning_rate * grad
    timer = time.time() - start
    total_time += timer
    print(f"Epoch: {i + 1:4d}/1000, cost: {mse:1.7f}, epoch time: {timer:.6f}s, average time: {total_time / (i + 1):.6f}", end="\r")
    
print()
    