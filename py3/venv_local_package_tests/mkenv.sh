#!/usr/bin/env bash

rm -rf .venv/ 
python3 -m venv .venv 
readonly sourceFile="./.venv/bin/activate"
source ${sourceFile} 
pip3 install -e .
#python3 setup.py develop
