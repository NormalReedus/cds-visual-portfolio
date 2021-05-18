#!/usr/bin/env bash

VENVNAME=assignment_1_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f requirements.txt && pip install -r requirements.txt

mkdir -p data/
mkdir -p output/

echo "build $VENVNAME"
