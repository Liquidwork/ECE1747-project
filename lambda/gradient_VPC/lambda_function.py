import json
import numpy as np
from data_process import decompress
import base64

def gradientCalculate(X, y, params):
    ''' Calculate the gradient for a batch of samples.
        - params: params for the linear regression. The last term is the constant for the
          linear regression model.
    '''
    
    X = np.concatenate((X, np.ones([X.shape[0], 1])), axis=1) # Concat column of 1 to the end of matrix X
    y = np.reshape(1, -1)
    params = np.array(params).reshape(1, -1)

    predict = (X.dot(params.T)) # Make prediction using current parameter.
    error = predict - y

    gradient = 2 * error.T.dot(X) / X.shape[0]
    
    mse = np.mean(error ** 2)
    return gradient.flatten(), mse

def lambda_handler(event, context):
    
    # {X:[[]], y:[], params:[]}
    try:
        event = base64.b64decode(event)
        X, y, params = decompress(event)
    except Exception as e:
        return {
        'statusCode': 400,
        'body': 'format not match\n' + str(e)
        }
    
    
    gradient, mse = gradientCalculate(X, y, params)
    
    
    return {
        'statusCode': 200,
        'body': json.dumps({"gradient": gradient.tolist(), "mse": mse})
    }
