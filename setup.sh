#!/bin/bash
if [ "$(uname)" = "Linux" ]; then
    PIPENV_PIPFILE=Pipfile.linux pipenv install
else
    PIPENV_PIPFILE=Pipfile.mac pipenv install
fi