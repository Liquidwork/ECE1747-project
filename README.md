# Cloud-based Serverless Parallel Gradient Descent

We are using the AWS Lambda function as the computation nodes, the Lambda has the auto-scaling feature that can automatically expand the number of processes to match the number of requests. We aim to achieve parallelization without being limited by hardware quantity.

## Project Structure

The project has 4 executables in total `local.py`, `mini-batch_gradient_descent.py`, `remote.py` and `boto.py`.

- `local.py`: Full-batch linear regression gradient descent running on the local machine. It will generate test data and start training the linear regression model. Time spent on each epoch and average time spent will be output.
- `mini-batch_gradient_descent.py`: The mini-batch gradient descent demo. It only has the local single-threaded version and will not output timing result.
- `remote.py`: Full-batch linear regression model using the AWS Lambda function as computation core. It uses HTTP protocol to communicate with the lambda function. The program will output the timing result of single epoch and average timing result. Does not need to deploy a Lambda function to run it.
- `boto.py`: Full-batch linear regression model using the AWS Lambda function as computation core. It uses boto3 package to invoke the lambda function. The program will output the timing result of single epoch and average timing result. Need to deploy a lambda function before running the code.

The project has 2 lambda function, `Gradient` and `gradient_VPC`.

- `Gradient`: Receive `json` format as input. Does not need to decompress data before calculation.
- `gradient_VPC`: Aimed to run inside a VPC, however, lambda function is not designed to be run inside any private network. Receive compressed data as input.

## How to run

Make sure `python3` and `numpy` package is correctly installed.

Run the code using command:

```shell
python3 program.py # substitute with correct filename
```

## Deploy the Lambda function

`boto.py` uses `boto3` package, which is needed to be deployed to the same account with the user. Deploy the lambda function `gradient_VPC` first to run it.
