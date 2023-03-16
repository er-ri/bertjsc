
## Deploy Endpoint on SageMaker(Real-time/Serverless Inference)
The folder lists some necessary files when deploying on AWS SageMaker Hosting Service. The jupyter notebook `SageMakerHosting.ipynb` includes a step-by-step instructions of deployment for both **real-time** and **serverless** endpoint types by AWS API.

## Note
### Creating `*.tar.gz`
1. Put the fine-tuned model under folder `/inference`.
2. `SageMakerHosting.ipynb`, a step-by-step sagemaker deployment instruction, can be executed on SageMaker Studio or Locally.
3. SageMaker Endpoint Hosting Service needs the `tar.gz` file followings the below folder structure, where `model.pth` is the fine-tuned model, `/code/inference.py` defines the model behavior(input, output and loading) and `/code/requirements.txt` is the dependencies, respectively.

Folder structure(`model.tar.gz`):
```
|- model.pth
|- code/
  |- inference.py
  |- requirements.txt  # only for versions 1.3.1 and higher
```
* Be careful when specifying packages' version in `requirements.txt`, some versions are not available on SageMaker. 

More details can be found at: 
https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#deploy-pytorch-models
 
---

### Invoke SageMaker using Lambda Function & API Gateway
* Call an Amazon SageMaker model endpoint using Amazon API Gateway and AWS Lambda, see [here](https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/).

