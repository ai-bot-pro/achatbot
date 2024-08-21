#!/bin/sh

if [[ $# -eq 0 ]] ; then
    echo "Usage: $0 S3_BUCKET <S3_PREFIX> <REGIONS>. Must provide at least an S3 bucket prefix."
    exit 1
fi

S3_BUCKET_PREFIX=$1
REGIONS=${2:-$(aws configure get region)}
IFS=',' read -r -a regionarr <<< "$REGIONS"
echo "Creating buckets with prefix: ${S3_BUCKET_PREFIX}"
echo "Region list is \"${REGIONS}\""

echo "Creating S3 code buckets"
for region in "${regionarr[@]}"
do
    echo "Creating bucket : \"${S3_BUCKET_PREFIX}-${region}\""
    aws s3 mb s3://${S3_BUCKET_PREFIX}-${region} --region ${region}
done
