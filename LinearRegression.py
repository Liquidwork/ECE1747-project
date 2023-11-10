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
    
# # Generate samples
# X1 = 2 * np.random.rand(1000000, 1)
# X2 = -2 * np.random.rand(1000000, 1)
# X = np.concatenate((X1, X2), axis=1)
# y = 4 + 3 * X1 + 2 * X2 + np.random.randn(1000000, 1)

# # Parameters for the regression
# params = [3, 4, 7]
# learning_rate = 0.05

# # Fit the model
# for i in range(1000):
#     grad, mse = gradientCalculate(X, y, params)
#     print(params, grad, mse)
#     # Change gradient according to learning rate
#     params -= learning_rate * grad
    
    