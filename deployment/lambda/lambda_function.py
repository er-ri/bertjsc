import os
import boto3
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    logger.info('Received event: %s', json.dumps(event, indent=2))
    data = json.loads(json.dumps(event))
    
    try:
        payload = data['text']
    except Exception as err:
        logger.error("%s", repr(err))
        return "Invalid data:" + repr(err)    

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)

    result = json.loads(response['Body'].read().decode('utf-8'))

    return result