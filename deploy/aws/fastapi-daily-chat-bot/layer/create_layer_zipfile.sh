#!/bin/sh
# !NOTE:
# CI with sam build with local os, if need complie, need use docker to do it
# or docker public.ecr.aws/sam/build-python${PY_VERSION}:latest-x86_64 to CI

set -e

docker=${1}
LAYER_ZIP_NAME="fastapi-daily-chat-bot-lambda-layer.zip"

# rm existing zipfiles
if [[ -f "${LAYER_ZIP_NAME}" ]]; then
    echo "Deleting zipfiles"
    rm *.zip
fi

echo "Installing packages"
cd ..

if [[ -n "${docker}" ]]; then
    # if want local test with local docker, open this
    DOCKER_HOST=unix://$HOME/.docker/run/docker.sock sam build -u
else
    # local build
    sam build
fi

echo "Bundling requirements.txt packages into zip file"
mkdir -p .aws-sam/build/layers
cp -R .aws-sam/build/FastapiDailyChatBotFunction/* .aws-sam/build/layers/
cd .aws-sam/build/layers/ 
du -sch . 

find . -type d -name "tests" -exec rm -rf {} +
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name "*.dist-info" -exec rm -rf {} +
find . -type d -name "*.egg-info" -exec rm -rf {} +
rm -rf ./{caffe2,wheel,pkg_resources,pip,pipenv,setuptools} 
#rm -f ./torch/lib/libtorch.so 
find . -name \*.pyc -delete
# sam package, codeUri from s3 bucket, so no need to package lambda handle code, NOTE: don't to sam build
rm {app.py,requirements.txt,__init__.py} 
du -sch . 
cd -
# build the .requirements.zip file
cd .aws-sam/build/layers/
zip -9 -q -r ../../../layer/.requirements.zip . 
cd -

echo "Creating layer zipfile"
cd layer
#unzip -l .requirements.zip
zip -9 ${LAYER_ZIP_NAME} .requirements.zip -r python/
unzip -l ${LAYER_ZIP_NAME}

echo "Delete the requirements zipfile & build directory"
rm .requirements.zip
rm -rf ../.aws-sam/build