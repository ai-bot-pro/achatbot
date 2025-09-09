#!/bin/bash
# bash scripts/pypi_achatbot.sh [dev|test|prod]

set -e 

pypi=
if [ $# -ge 1 ] && [ "$1" == "prod" ]; then
    pypi=pypi
    echo "upload to $pypi"
fi
if [ $# -ge 1 ] && [ "$1" == "test" ]; then
    pypi=testpypi
    echo "upload to $pypi"
fi
if [ $# -ge 1 ] && [ "$1" == "dev" ]; then
    pypi=devpypi
    echo "install to local package dev"
fi

rm -rf pypi_build/app/achatbot
mkdir -p pypi_build/app/achatbot

cp -r src/* pypi_build/app/achatbot/
rm -f deps/CosyVoice/third_party/Matcha-TTS/data
rm -f deps/GLM4Voice/third_party/Matcha-TTS/data
rm -f deps/KimiAudio/kimia_infer/models/tokenizer/glm4/third_party/Matcha-TTS/data
rm -f deps/VITAAudio/third_party/GLM-4-Voice/third_party/Matcha-TTS/data


echo "copy deps ..."
#cp -r deps/* pypi_build/app/achatbot/
find deps -type f -name "*.py" -exec sh -c '
    dest="pypi_build/app/achatbot/$(echo {} | sed "s|^deps/||")"
    mkdir -p "$(dirname "$dest")"
    cp {} "$dest"
' \;



echo "replace ..."
find pypi_build/app/achatbot/ | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

find pypi_build/app/achatbot/ -type f -print0 | xargs -0 perl -i -pe \
  's/from src\./from achatbot\./g;s/from deps\./from achatbot\./g;s/import src\./import achatbot\./g;s/import deps\./import achatbot\./g'


if [ -n "$pypi" ]; then
    echo "build ..."
    pip install -q build
    rm -rf dist && python3 -m build 
    if [ "$pypi" == "devpypi" ]; then
        echo "install ..."
        pip install -U dist/*.whl
    else
        echo "upload ..."
        twine upload --verbose --skip-existing --repository $pypi dist/*
    fi
    echo "clear ..."
    rm -rf pypi_build/app/achatbot
fi 

