#!/bin/bash
if [ "$(uname)" = "Linux" ]; then
    pipenv run pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu130 --no-deps
fi
pipenv install