#!/bin/bash
# bash scripts/pypi_achatbot.sh [dev|test|prod]

set -e 

pypi=
if [ $# -ge 1 ] && [ "$1" == "prod" ]; then
    pypi=pypi
fi
if [ $# -ge 1 ] && [ "$1" == "test" ]; then
    pypi=testpypi
fi
if [ $# -ge 1 ] && [ "$1" == "dev" ]; then
    pypi=devpypi
fi
echo "upload to $pypi"

rm -rf pypi_build/app/achatbot
mkdir -p pypi_build/app/achatbot

cp -r src/* pypi_build/app/achatbot/
rm -f deps/CosyVoice/third_party/Matcha-TTS/data
rm -f deps/GLM4Voice/third_party/Matcha-TTS/data
rm -f deps/KimiAudio/kimia_infer/models/tokenizer/glm4/third_party/Matcha-TTS/data
rm -f deps/VITAAudio/third_party/GLM-4-Voice/third_party/Matcha-TTS/data
cp -r deps/* pypi_build/app/achatbot/


find pypi_build/app/achatbot/ | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

find pypi_build/app/achatbot/ -type f -print0 | xargs -0 perl -i -pe \
  's/from src\./from achatbot\./g;s/from deps\./from achatbot\./g;s/import src\./import achatbot\./g;s/import deps\./import achatbot\./g'


if [ -n "$pypi" ]; then
    pip install -q build
    rm -rf dist && python3 -m build 
    if [ "$pypi" == "devpypi" ]; then
        pip install -U dist/*.whl
    else
        twine upload --verbose --skip-existing --repository $pypi dist/*
    fi
    rm -rf pypi_build/app/achatbot
fi 

