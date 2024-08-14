#!/bin/bash
# bash scripts/pipy_achatbot.sh

mkdir -p pipy_build/app/achatbot

cp -r src/* pipy_build/app/achatbot/
cp -r deps/* pipy_build/app/achatbot/

find pipy_build/app/achatbot/ | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

find pipy_build/app/achatbot/ -type f -print0 | xargs -0 perl -i -pe 's/src\./achatbot\./g'
find pipy_build/app/achatbot/ -type f -print0 | xargs -0 perl -i -pe 's/deps\./achatbot\./g'

rm -rf dist && python3 -m build 

twine upload --verbose --skip-existing --repository testpypi  dist/*

twine upload --verbose --skip-existing --repository pypi  dist/*