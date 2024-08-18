#!/bin/sh

set -e

if [[ $# -eq 0 ]] ; then
    echo "Usage: $0 S3_BUCKET_PREFIX [<S3_OBJECT_PREFIX> <LAYER_NAME> <REGIONS> <PYTORCH_VERSION> <LAYER_VERSION>]."
    echo "Must provide at least an S3 bucket prefix."
    exit 1
fi

# process the input parameters
S3_BUCKET_PREFIX=$1
S3_OBJECT_PREFIX=${2:-lambda-layers}
LAYER_NAME=${3:-fastapi-daily-chat-bot-p311}
REGIONS=${4:-$(aws configure get region)}
LAYER_VERSION=${6:-2}
IFS=',' read -r -a regionarr <<< "$REGIONS"

LAYER_ZIP_NAME="fastapi-daily-chat-bot-lambda-layer.zip"
echo "Using S3 Bucket prefix ${S3_BUCKET_PREFIX}"
echo "S3 Prefix ${S3_OBJECT_PREFIX}"
echo "Lambda Layer name is ${LAYER_NAME}"q
echo "Layer Zip ${LAYER_ZIP_NAME}"
echo "Region list is ${REGIONS}"

if [[ ! -f "${LAYER_ZIP_NAME}" ]]; then
    echo "Create zipfile first by running  the script ./create_layer_zipfile.sh <LAYER_ZIP_NAME>"
    exit 1
fi

echo "Creating Lambda Layers"
for region in "${regionarr[@]}"
do
    echo "Uploading zip to S3 bucket \"${S3_BUCKET_PREFIX}-${region}\""
    aws s3 sync . s3://${S3_BUCKET_PREFIX}-${region}/${S3_OBJECT_PREFIX}/ --exclude "*" --include "${LAYER_ZIP_NAME}"

    echo "Creating Lambda layer in region: $region"
    aws lambda publish-layer-version \
        --layer-name "${LAYER_NAME}" \
        --description "Lambda layer of PyTorch ${PYTORCH_VERSION} zipped to be extracted with unzip_requirements file" \
        --content "S3Bucket=${S3_BUCKET_PREFIX}-${region},S3Key=${S3_OBJECT_PREFIX}/${LAYER_ZIP_NAME}" \
        --compatible-runtimes "python3.11" \
        --license-info "BSD-3-Clause" \
        --region "${region}"

    echo "Creating lambda layer permissions"
    aws lambda add-layer-version-permission \
        --layer-name "${LAYER_NAME}" \
        --version-number ${LAYER_VERSION} \
        --statement-id "public-access" \
        --action "lambda:GetLayerVersion" \
        --principal "*" \
        --region "${region}"
done
