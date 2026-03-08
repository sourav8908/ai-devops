# test_lambda.py
from lambda_function import lambda_handler

# Simulate API Gateway request
test_event = {
    "question": "What AWS services does DIBS use?"
}

result = lambda_handler(test_event, None)
print(result)