import requests
import numpy as np
import json
import time

url = "https://kquqqi3ijnocedkbxqhyoqbfoa0bjpca.lambda-url.us-east-1.on.aws/Gradient"

# Generate samples

X1 = 2 * np.random.rand(1000, 1)
X2 = -2 * np.random.rand(1000, 1)
X = np.concatenate((X1, X2), axis=1)
y = 4 + 3 * X1 + 2 * X2 + np.random.randn(1000, 1)

# Parameters for the regression
params = [3., 4., 7.]
learning_rate = 0.05

X = X.tolist()
y = y.tolist()
params = np.asarray(params)

body = json.dumps({"X": X, "y": y}) # This part will not change.
# print(body)
body = body[:-1] + ', "params": '

total_time = 0

# Fit the model
for i in range(1000):
    start = time.time()
    data = body + str(params.tolist()) + "}"
    r = requests.post(url, data=data, headers={"Content-Type": "application/json"})
    
    if r.status_code==200:
        # Fit the model
        res = r.json()
        grad = res["gradient"]
        mse = res["mse"]
        # print(params, grad, mse)
        timer = time.time() - start
        total_time += timer
        print(f"Epoch: {i:4d}/1000, cost: {mse:1.7f}, epoch time: {timer:.6f}s, average time: {total_time / (i + 1):.6f}", end="\r")
        # Change gradient according to learning rate
        params -= learning_rate * np.asarray(grad)
    else: 
        print(r.status_code)
        print(r.text)

print()