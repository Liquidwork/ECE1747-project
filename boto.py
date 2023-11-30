import requests
import numpy as np
import json
import time
import boto3
import base64
from data_process import compress

url = "https://kquqqi3ijnocedkbxqhyoqbfoa0bjpca.lambda-url.us-east-1.on.aws/Gradient"

# Generate samples

client = boto3.client('lambda')

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
    data = base64.b64encode(compress(X, y, params))
    r = client.invoke(
        FunctionName='gradient_VPC',
        InvocationType='RequestResponse',
        LogType='None',
        Payload=json.dumps(data.decode("ascii"))
    )
    
    # print(r)
    
    if r['ResponseMetadata']['HTTPStatusCode'] == 200:
        # Fit the model
        
        result = str(r["Payload"].read())[31:-3]
        result = result.replace("\\", "")
        
        # print(result)
        res = json.loads(result)
        grad = res["gradient"]
        mse = res["mse"]
        # print(params, grad, mse)
        timer = time.time() - start
        total_time += timer
        print(f"Epoch: {i + 1:4d}/1000, cost: {mse:1.7f}, epoch time: {timer:.6f}s, average time: {total_time / (i + 1):.6f}", end="\r")
        # Change gradient according to learning rate
        params -= learning_rate * np.asarray(grad)
    else: 
        print(r['ResponseMetadata']['HTTPStatusCode'])

print()