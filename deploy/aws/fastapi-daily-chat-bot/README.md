# Deploy FastAPI on AWS: Lambda + API Gateway

This sample deploys FastAPI Daily Chat Bots in a Lambda function that is fronted by an HTTP API in API Gateway.

## Requirements
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
- An S3 bucket

## Run Locally

The SAM CLI [lets you run APIs locally](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-using-start-api.html).


### Build the application
build with pip install requirements.txt, need some time to install dependencies

```bash
$ sam build

...

Build Succeeded

Built Artifacts  : .aws-sam/build
Built Template   : .aws-sam/build/template.yaml

Commands you can use next
=========================
[*] Invoke Function: sam local invoke
[*] Deploy: sam deploy --guided
```

### Start the API locally 
need local install docker and run it with img public.ecr.aws/lambda/python:3.11-x86_64

```bash
$ sam local invoke
$ sam local start-api

Mounting Function at http://127.0.0.1:3000$default [X-AMAZON-APIGATEWAY-ANY-METHOD]
You can now browse to the above endpoints to invoke your functions.

2024-08-12 22:20:56  * Running on http://127.0.0.1:3000/ (Press CTRL+C to quit)
```

### Call the API
```bash
$ curl http://127.0.0.1:3000

{"message":"FastAPI running in a Lambda function"}
```

## Deploy

The SAM CLI is also used to deploy the application.

### Build

```bash
$ sam build
```

### Deploy

```bash
$ sam deploy --s3-bucket YOUR_BUCKET_NAME
...

CloudFormation outputs from deployed stack
---------------------------------------------------------------------------
Outputs
---------------------------------------------------------------------------
Key                 ApiUrl
Description         URL of your API endpoint
Value               https://API_ID.execute-api.eu-west-1.amazonaws.com/
---------------------------------------------------------------------------

Successfully created/updated stack - FastAPIOnLambda in eu-west-1
```

### Call the API

The URL of your API Gateway is shown in the outputs section of the `sam deploy` command.

```bash
$ curl https://API_ID.execute-api.eu-west-1.amazonaws.com/

{"message":"FastAPI running in a Lambda function"}
```