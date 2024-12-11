#/bin/bash

#pip install autopep8
#autopep8 -a -a --max-line-length=100 --in-place -r .

pip install ruff
ruff format --config pyproject.toml
ruff check --fix --config pyproject.toml