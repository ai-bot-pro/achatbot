# Deploy FastAPI + Daily + Chat Bots on AWS: Lambda + API Gateway

This sample deploys FastAPI Daily Chat Bots in a Lambda function that is fronted by an HTTP API in API Gateway.
> [!NOTE]
> daily python sdk install issue see this [issue](https://github.com/daily-co/daily-python/issues/25)

## Requirements
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
- An S3 bucket
- Docker (if local test, need pull img to deploy `sam build -u`)

## Run Locally

The SAM CLI [lets you run APIs locally](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-using-start-api.html).

### Local development

**Creating test Lambda Environment Variables**

First create a file called `env.json` with the payload similar to the following substituting the values for the S3 Bucket and Key where your PyTorch model has been saved on S3.

```json
{
    "PyTorchFunction": {
      "MODEL_BUCKET": "REPLACE_THIS_WITH_YOUR_MODEL_S3_BUCKET_NAME",  
    }
}
```

**Invoking function locally using a local sample payload**

Edit the file named `event.json` and enter a value for the JSON value `url` to the image you want to classify.

Call the following sam command to test the function locally.

```bash
sam local invoke PyTorchFunction -n env.json -e event.json
```

**Invoking function locally through local API Gateway**

```bash
sam local start-api -n env.json
```

If the previous command ran successfully you should now be able to send a post request to the local endpoint.

An example is the following:

```bash
curl -H "Content-Type: application/json" \
    -X POST http://localhost:3000/
```

## Deploy Setup process
> [!NOTE]
> sam package, build, deploy params can set in samconfig.toml
> more detail see: 
> https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-config.html

### Create S3 Bucket
First, we need a S3 bucket where we can upload our Lambda functions and layers packaged as ZIP files before we deploy anything - If you don't have a S3 bucket to store code artifacts then this is a good time to create one, e.g.:
```bash
aws s3 mb s3://chat-bots-bucket 
```
or in layer dir run `sh ./create_code_buckets.sh` to create code bucket with region

### Packaging
Next, run the following command to package our Lambda function to S3:
```bash
sam package \
    --output-template-file packaged.yaml \
    --s3-bucket chat-bots-bucket
```

### Deploy SAM resources with un build deps packages
Next, the following command will create a Cloudformation Stack and deploy your SAM resources. You will need to override the default parameters for the bucket name and key. This is done by passing the --parameter-overrides option to the deploy command.
```bash
sam deploy \
    --template-file packaged.yaml \
    --parameter-overrides BucketName=chat-bots-bucket

...

Key                 ApiUrl                                                                                                                                                
Description         URL of your API endpoint                                                                                                                              
Value               https://880gye3rqj.execute-api.us-east-1.amazonaws.com/                                                                                               

Key                 FastapiDailyChatBotFunciton                                                                                                                           
Description         Fastapi Daily Chat Bot Lambda Function ARN                                                                                                            
Value               arn:aws:lambda:us-east-1:139573341397:function:FastapiDailyChatBotLambda                                                                              

Key                 FastapiDailyChatBotFuncitonIamRole                                                                                                                    
Description         Implicit IAM Role created for Fastapi Daily Chat Bot Lambda Function                                                                                  
Value               arn:aws:iam::139573341397:role/fastapi-daily-chat-bot-FastapiDailyChatBotFunctionR-SgsarUQBJ2c7   
```
> [!NOTE]
> **See [Serverless Application Model (SAM) HOWTO Guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-quick-start.html) for more details in how to get started.**

After deployment is complete you can run the following command to retrieve the API Gateway Endpoint URL:
```bash
aws cloudformation describe-stacks \
    --stack-name fastapi-daily-chat-bot \
    --query 'Stacks[].Outputs[?OutputKey==`ApiUrl`]' \
    --output table

# see more detail
aws cloudformation describe-stacks \
    --stack-name fastapi-daily-chat-bot --output table
```
so u can access the api gateway endpoint url by `https://<api-id>.execute-api.<region>.amazonaws.com/<stage>/<path>`
> [!TIPS]
> becuase don't to build app, some deps don't to deploy, so access error,like this:
> ```
> {
>   message: "Internal Server Error"
> }
> ```
> u can use `aws logs` to filter logs, see blow

### Fetch, tail, and filter Lambda function logs

To simplify troubleshooting, SAM CLI has a command called sam logs. sam logs lets you fetch logs generated by your Lambda function from the command line. In addition to printing the logs on the terminal, this command has several nifty features to help you quickly find the bug.

> [!NOTE]
> This command works for all AWS Lambda functions; not just the ones you deploy using SAM.

```bash
sam logs -n FastapiDailyChatBotFunction --stack-name fastapi-daily-chat-bot --tail

2024/08/18/[$LATEST]60cefc4c4d174d898e4dbb301f9f10ef 2024-08-18T11:04:10.079000 INIT_START Runtime Version: python:3.11.v39     Runtime Version ARN: arn:aws:lambda:us-east-1::runtime:93e1685a33effc0cd2497a9a132c7328deb2e03b8a600ecae522c00fc95a1c8f
2024/08/18/[$LATEST]60cefc4c4d174d898e4dbb301f9f10ef 2024-08-18T11:04:11.524000 [ERROR] Runtime.ImportModuleError: Unable to import module 'app': No module named 'daily'
Traceback (most recent call last):
2024/08/18/[$LATEST]60cefc4c4d174d898e4dbb301f9f10ef 2024-08-18T11:04:11.750000 INIT_REPORT Init Duration: 1670.77 ms   Phase: init     Status: error   Error Type: Runtime.Unknown
2024/08/18/[$LATEST]60cefc4c4d174d898e4dbb301f9f10ef 2024-08-18T11:04:13.241000 [ERROR] Runtime.ImportModuleError: Unable to import module 'app': No module named 'daily'
Traceback (most recent call last):
2024/08/18/[$LATEST]60cefc4c4d174d898e4dbb301f9f10ef 2024-08-18T11:04:13.484000 INIT_REPORT Init Duration: 1712.41 ms   Phase: invoke   Status: error   Error Type: Runtime.Unknown
2024/08/18/[$LATEST]60cefc4c4d174d898e4dbb301f9f10ef 2024-08-18T11:04:13.484000 START RequestId: c16b0a58-61af-4b5e-ae57-b5dad549e389 Version: $LATEST
2024/08/18/[$LATEST]60cefc4c4d174d898e4dbb301f9f10ef 2024-08-18T11:04:13.485000 Unknown application error occurred
Runtime.Unknown
2024/08/18/[$LATEST]60cefc4c4d174d898e4dbb301f9f10ef 2024-08-18T11:04:13.485000 END RequestId: c16b0a58-61af-4b5e-ae57-b5dad549e389
2024/08/18/[$LATEST]60cefc4c4d174d898e4dbb301f9f10ef 2024-08-18T11:04:13.485000 REPORT RequestId: c16b0a58-61af-4b5e-ae57-b5dad549e389  Duration: 1713.36 ms    Billed Duration: 1714 ms      Memory Size: 3008 MB    Max Memory Used: 102 MB
```
> [!TIPS]
> You can find more information and examples about filtering Lambda function logs in the [SAM CLI Documentation](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-logging.html).

### Testing

Next, we install test dependencies and we run `pytest` against our `tests` folder to run our initial unit tests:

```bash
pip install pytest pytest-mock
python -m pytest tests/ -v
```

### Cleanup

In order to delete our Serverless Application recently deployed you can use the following AWS CLI Command:

```bash
aws cloudformation delete-stack --stack-name fastapi-daily-chat-bot
```

## Build the application with Lambda Lyaer
Have shown how to create a SAM application to do PyTorch model inference. Now you will learn how to create your own Lambda Layer to package the PyTorch dependencies.

### Creating custom Lambda Layer
The project uses [Lambda layers](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html) for deploying the PyTorch libraries. **Lambda Layers** allow you to bundle dependencies without needing to include them in your application bundle.

The project defaults to using a public Lambda Layer ARN `arn:aws:lambda:us-east-1:139573341397:layer:fastapi-daily-chat-bot-p311:1` containing the PyTorch v2.2.2 packages. It is publically accessible. To build and publish your own PyTorch layer follow the instuctions below.

> [!NOTE]
> AWS Lambda has a limit of 250 MB for the deployment package size including lamba layers. 
> PyTorch plus its dependencies is more than this so we need to implement a trick to get around this limit. 

We will create a zipfile called `.requirements.zip` with all the PyTorch and associated packages. We will then add this zipfile to the Lambda Layer zipfile along with a python script called `unzip_requirements.py`. The python script will extract the zipfile `.requirements.zip` to the `/tmp` when the Lambda execution context is created. 

#### Layer creation steps

Goto the directory named `layer` and run the script named `sh ./create_layer_zipfile.sh`. This will launch the command `sam build --use-container` to download the packages defined in the `requirements.txt` file. The script will remove unncessary files and directories and then create the zipfile `.requirements.zip` then bundle this zipfile with the python script `unzip_requirements.py` to the zipfile `fastapi-daily-chat-bot-lambda-layer.zip`.

```bash
cd layer
sh ./create_layer_zipfile.sh
```
Upload the Lambda Layer zipfile to one of your S3 buckets. Take note of the S3 URL as it will be used when creating the Lambda Layer.

```bash
aws s3 cp fastapi-daily-chat-bot-lambda-layer.zip s3://chat-bots-bucket/lambda-layers/fastapi-daily-chat-bot-lambda-layer.zip
```
Now we can create the Lambda Layer version. Execute the following AWS CLI command:

```bash
aws lambda publish-layer-version \
    --layer-name "fastapi-daily-chat-bot-p311" \
    --description "Lambda layer of fastapi chat bot zipped to be extracted with unzip_requirements file" \
    --content "S3Bucket=chat-bots-bucket,S3Key=lambda-layers/fastapi-daily-chat-bot-lambda-layer.zip" \
    --compatible-runtimes "python3.11" 
```

Take note of the value of the response parameter `LayerVersionArn`. 

or in layer dir run `sh ./publish_lambda_layers.sh`

### Local Docker Testing Lambda Layer

The following examples show how you can use your own Lambda Layer in both local testing and then deploying to AWS. They will overide the default Lambda Layer in the file `template.yaml`.

**Invoking function locally overriding Lambda Layer default**

```bash
sam local invoke FastapiDailyChatBotFunction -n env.json -e event.json \
    --parameter-overrides LambdaLayerArn=REPLACE_WITH_YOUR_LAMBDA_LAYER_ARN
```

**Invoking function through local API Gateway overriding Lambda Layer default**

```bash
sam local start-api -n env.json \
    --parameter-overrides LambdaLayerArn=REPLACE_WITH_YOUR_LAMBDA_LAYER_ARN
```

### Deploying Lambda Layer

**Deploying the Lambda function overriding Lambda Layer default**

```bash
sam deploy \
    --template-file packaged.yaml \
    --parameter-overrides LambdaLayerArn=REPLACE_WITH_YOUR_LAMBDA_LAYER_ARN BucketName=chat-bots-bucket
```

### Lambda code format

At the beginning of the file `runtime/app.py` you need to include the following code that will unzip the package file containing the python libs. It will extract the package zip file named `.requirements.zip` to the `/tmp` to get around the unzipped Lambda deployment package limit of 250 MB.

```python
try:
    import unzip_requirements
except ImportError:
    pass
```

After these lines you can import all the python libraries you need to.

# Reference
- https://serverlessland.com/learn
- https://github.com/aws/aws-sam-cli
- https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-config.html
- https://github.com/aws/aws-lambda-builders
- https://segments.ai/blog/pytorch-on-lambda/ (PyTorch on AWS Lambda)
- https://github.com/JTunis/create-pytorch-lambda-layer (create PyTorch lambda layer, is older,no images to run container, should use `sam build -u` to build layer, then use sls , TF or sam/cdk to deploy)
- https://github.com/mattmcclean/sam-pytorch-example.git (**sam deploy PyTorch on AWS Lambda**)
- https://github.com/weedge/craftsman/tree/main/cloud/aws/cdk/serverless-openai-chatbot (**serverless openai chatbot**)