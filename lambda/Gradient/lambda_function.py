import json
import numpy as np

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
    
    # Data will be in form of
    event = json.loads(event['body'])
    # {X:[[]], y:[], params:[]}
    try:
        X = np.asarray(event["X"])
        y = np.asarray(event["y"])
        params = np.asarray(event["params"])
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
