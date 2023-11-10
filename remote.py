import requests
import numpy as np

url = "https://kquqqi3ijnocedkbxqhyoqbfoa0bjpca.lambda-url.us-east-1.on.aws/Gradient"

# Generate samples

X1 = 2 * np.random.rand(1000, 1)
X2 = -2 * np.random.rand(1000, 1)
X = np.concatenate((X1, X2), axis=1)
y = 4 + 3 * X1 + 2 * X2 + np.random.randn(1000, 1)

# Parameters for the regression
params = [[3., 4., 7.]]
learning_rate = 0.05

X = X.tolist()
y = y.tolist()
params = np.asarray(params)

body={"X": X, "y": y, "params": params}

# Fit the model
for i in range(1000):
    
    body["params"] = params.tolist()
    r = requests.post(url, json=body)
    
    if r.status_code==200:
        # Fit the model
        res = r.json()
        grad = res["gradient"]
        mse = res["mse"]
        print(params, grad, mse)
        # Change gradient according to learning rate
        params -= learning_rate * np.asarray(grad)
    else: print(r.status_code)
